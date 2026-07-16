function run_nprobe_sweep(
    context::BenchmarkContext,
    vectors::AbstractMatrix,
    metadata::AbstractVector,
    queries::AbstractMatrix,
    filters::AbstractVector,
    nprobes::AbstractVector{<:Integer};
    k::Int = 10,
    repetitions::Int = 3,
    metric::Symbol = :cosine,
    postfilter_oversample::Int = 10,
    adaptive_postfilter::Bool = true,
    adaptive::Bool = false,
    max_nprobe::Union{Nothing,Int} = nothing,
    candidate_multiplier::Float64 = 4.0,
    vector_weight::Float64 = 0.5,
    filter_weight::Float64 = 0.5,
    rerank_factor::Int = 4,
)
    isempty(nprobes)&&throw(ArgumentError("nprobes cannot be empty"))

    list_count=length(context.index.ivf.lists)
    probe_values=sort!(unique(Int.(nprobes)))
    all(probe->1<=probe<=list_count, probe_values)||throw(
        ArgumentError("nprobe must be between 1 and list count"),
    )
    resolved_max_nprobe=max_nprobe===nothing ? maximum(probe_values) : max_nprobe
    1<=resolved_max_nprobe<=list_count||throw(
        ArgumentError("max nprobe must be between 1 and list count"),
    )

    return [
        (
            nprobe = nprobe,
            benchmark = run_benchmark(
                context,
                vectors,
                metadata,
                queries,
                filters;
                k = k,
                nprobe = nprobe,
                repetitions = repetitions,
                metric = metric,
                postfilter_oversample = postfilter_oversample,
                adaptive_postfilter = adaptive_postfilter,
                adaptive = adaptive,
                max_nprobe = max(resolved_max_nprobe, nprobe),
                candidate_multiplier = candidate_multiplier,
                vector_weight = vector_weight,
                filter_weight = filter_weight,
                rerank_factor = rerank_factor,
            ),
        ) for nprobe in probe_values
    ]
end

function run_nprobe_sweep(
    vectors::AbstractMatrix,
    metadata::AbstractVector,
    queries::AbstractMatrix,
    filters::AbstractVector,
    nprobes::AbstractVector{<:Integer};
    nlists::Int,
    k::Int = 10,
    iterations::Int = 20,
    seed::Int = 42,
    repetitions::Int = 3,
    metric::Symbol = :cosine,
    restarts::Int = 1,
    training_count::Int = size(vectors, 2),
    postfilter_oversample::Int = 10,
    adaptive_postfilter::Bool = true,
    adaptive::Bool = false,
    max_nprobe::Union{Nothing,Int} = nothing,
    candidate_multiplier::Float64 = 4.0,
    vector_weight::Float64 = 0.5,
    filter_weight::Float64 = 0.5,
    rerank_factor::Int = 4,
)
    context=build_benchmark_context(
        vectors,
        metadata;
        nlists = nlists,
        iterations = iterations,
        seed = seed,
        metric = metric,
        restarts = restarts,
        training_count = training_count,
    )

    return run_nprobe_sweep(
        context,
        vectors,
        metadata,
        queries,
        filters,
        nprobes;
        k = k,
        repetitions = repetitions,
        metric = metric,
        postfilter_oversample = postfilter_oversample,
        adaptive_postfilter = adaptive_postfilter,
        adaptive = adaptive,
        max_nprobe = max_nprobe,
        candidate_multiplier = candidate_multiplier,
        vector_weight = vector_weight,
        filter_weight = filter_weight,
        rerank_factor = rerank_factor,
    )
end

function sweep_points(sweep_results::AbstractVector, method::Symbol)
    points=NamedTuple[]

    for sweep_result in sweep_results
        hasproperty(sweep_result.benchmark, method)||continue
        result=getproperty(sweep_result.benchmark, method)
        result===nothing&&continue
        push!(
            points,
            (
                method = method,
                nprobe = sweep_result.nprobe,
                recall = result.average_recall,
                p50_ms = result.latency.p50_ms,
                p95_ms = result.latency.p95_ms,
                candidates_visited = result.average_candidates_visited,
                candidates_scored = result.average_candidates_scored,
                lists_probed = result.average_lists_probed,
            ),
        )
    end

    return points
end

function pareto_frontier(sweep_results::AbstractVector, method::Symbol)
    points=sweep_points(sweep_results, method)
    sort!(points; by = point->(point.p50_ms, -point.recall, point.candidates_scored))

    frontier=NamedTuple[]
    best_recall=-Inf

    for point in points
        if point.recall>best_recall
            push!(frontier, point)
            best_recall=point.recall
        end
    end

    return frontier
end

function best_at_recall(
    sweep_results::AbstractVector,
    method::Symbol,
    target_recall::Float64,
)
    0.0<=target_recall<=1.0||throw(ArgumentError("target recall must be between 0 and 1"))
    eligible=[
        point for
        point in sweep_points(sweep_results, method) if point.recall>=target_recall
    ]
    isempty(eligible)&&return nothing

    sort!(eligible; by = point->(point.p50_ms, point.candidates_scored, point.nprobe))
    return first(eligible)
end

function conservative_at_recall(
    sweep_results::AbstractVector,
    method::Symbol,
    target_recall::Float64;
    probe_safety_factor::Float64 = 2.0,
)
    probe_safety_factor>=1.0||throw(ArgumentError("probe safety factor must be atleast 1"))
    selected=best_at_recall(sweep_results, method, target_recall)
    selected===nothing&&return nothing
    target_nprobe=ceil(Int, selected.nprobe*probe_safety_factor)
    points=sort(sweep_points(sweep_results, method); by = point->point.nprobe)
    candidates=[point for point in points if point.nprobe>=target_nprobe]
    return isempty(candidates) ? last(points) : first(candidates)
end

function compare_at_recall(
    sweep_results::AbstractVector,
    methods::AbstractVector{Symbol},
    target_recall::Float64,
)
    return [
        (method = method, result = best_at_recall(sweep_results, method, target_recall)) for
        method in methods
    ]
end
