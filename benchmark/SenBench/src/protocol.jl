struct ExperimentSpec
    name::String
    vector_count::Int
    dimension::Int
    train_query_count::Int
    heldout_query_count::Int
    nlists::Int
    nprobes::Vector{Int}
    k::Int
    target_recall::Float64
    selection_margin::Float64
    probe_safety_factor::Float64
    minimum_speedup::Float64
    recall_tolerance::Float64
    candidate_multiplier::Float64
    postfilter_multipliers::Vector{Float64}
    postfilter_safety_factor::Float64
    vector_weight::Float64
    filter_weight::Float64
    rerank_factor::Int
    repetitions::Int
    iterations::Int
    restarts::Int
    training_count::Int
    metric::Symbol
    vector_workload::Symbol
    filter_workload::Symbol
    selectivity::Float64
    seed::Int
end

function ExperimentSpec(
    name::AbstractString;
    vector_count::Int,
    dimension::Int,
    train_query_count::Int = 20,
    heldout_query_count::Int = 50,
    nlists::Int = min(64, vector_count),
    nprobes::AbstractVector{<:Integer} = [1, 2, 4, 8, 16, 32, 64],
    k::Int = 10,
    target_recall::Float64 = 0.95,
    selection_margin::Float64 = 0.02,
    probe_safety_factor::Float64 = 2.0,
    minimum_speedup::Float64 = 1.15,
    recall_tolerance::Float64 = 0.01,
    candidate_multiplier::Float64 = 8.0,
    postfilter_multipliers::AbstractVector{<:Real} = [4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
    postfilter_safety_factor::Float64 = 2.0,
    vector_weight::Float64 = 0.5,
    filter_weight::Float64 = 0.5,
    rerank_factor::Int = 4,
    repetitions::Int = 3,
    iterations::Int = 10,
    restarts::Int = 2,
    training_count::Int = min(vector_count, 20_000),
    metric::Symbol = :cosine,
    vector_workload::Symbol = :gaussian,
    filter_workload::Symbol = :random,
    selectivity::Float64 = 0.05,
    seed::Int = 42,
)
    vector_count>0||throw(ArgumentError("vector count must be positive"))
    dimension>0||throw(ArgumentError("dimension must be positive"))
    train_query_count>0||throw(ArgumentError("train query count must be positive"))
    heldout_query_count>0||throw(ArgumentError("heldout query count must be positive"))
    1<=nlists<=vector_count||throw(
        ArgumentError("nlists must be between 1 and vector count"),
    )
    k>0||throw(ArgumentError("k must be positive"))
    0.0<target_recall<=1.0||throw(ArgumentError("target recall must be between 0 and 1"))
    selection_margin>=0||throw(ArgumentError("selection margin cannot be negative"))
    probe_safety_factor>=1.0||throw(ArgumentError("probe safety factor must be atleast 1"))
    minimum_speedup>1||throw(ArgumentError("minimum speedup must be greater than 1"))
    recall_tolerance>=0||throw(ArgumentError("recall tolerance cannot be negative"))
    candidate_multiplier>0||throw(ArgumentError("candidate multiplier must be positive"))
    multipliers=sort!(unique(Float64.(postfilter_multipliers)))
    isempty(multipliers)&&throw(ArgumentError("postfilter multipliers cannot be empty"))
    all(multiplier->multiplier>0, multipliers)||throw(
        ArgumentError("postfilter multipliers must be positive"),
    )
    postfilter_safety_factor>=1.0||throw(
        ArgumentError("postfilter safety factor must be atleast 1"),
    )
    vector_weight>=0||throw(ArgumentError("vector weight cannot be negative"))
    filter_weight>=0||throw(ArgumentError("filter weight cannot be negative"))
    vector_weight+filter_weight>0||throw(
        ArgumentError("atleast one list weight must be positive"),
    )
    rerank_factor>0||throw(ArgumentError("rerank factor must be positive"))
    repetitions>0||throw(ArgumentError("repetitions must be positive"))
    iterations>0||throw(ArgumentError("iterations must be positive"))
    restarts>0||throw(ArgumentError("restarts must be positive"))
    nlists<=training_count<=vector_count||throw(
        ArgumentError("training count must be between nlists and vector count"),
    )
    metric in (:cosine, :dot)||throw(ArgumentError("metric must be cosine or dot"))
    vector_workload in (:gaussian, :clustered, :real)||throw(
        ArgumentError("invalid vector workload"),
    )
    filter_workload in (:random, :correlated, :anticorrelated, :skewed, :natural)||throw(
        ArgumentError("invalid filter workload"),
    )
    0.0<=selectivity<=1.0||throw(ArgumentError("selectivity must be between 0 and 1"))

    probes=sort!(unique(Int.(nprobes)))
    isempty(probes)&&throw(ArgumentError("nprobes cannot be empty"))
    all(probe->1<=probe<=nlists, probes)||throw(
        ArgumentError("nprobe must be between 1 and nlists"),
    )

    return ExperimentSpec(
        String(name),
        vector_count,
        dimension,
        train_query_count,
        heldout_query_count,
        nlists,
        probes,
        k,
        target_recall,
        selection_margin,
        probe_safety_factor,
        minimum_speedup,
        recall_tolerance,
        candidate_multiplier,
        multipliers,
        postfilter_safety_factor,
        vector_weight,
        filter_weight,
        rerank_factor,
        repetitions,
        iterations,
        restarts,
        training_count,
        metric,
        vector_workload,
        filter_workload,
        selectivity,
        seed,
    )
end

function experiment_methods(spec::ExperimentSpec)
    methods=[:exact, :ivf_prefilter, :ivf_postfilter, :filter_aware]
    spec.metric===:cosine&&push!(methods, :filter_aware_bound)
    return methods
end

function generate_experiment_base(spec::ExperimentSpec)
    spec.vector_workload===:real&&throw(
        ArgumentError("real workloads require an external dataset"),
    )

    if spec.vector_workload===:clustered
        dataset=generate_clustered_dataset(
            spec.vector_count,
            spec.dimension;
            clusters = spec.nlists,
            cluster_noise = 0.08,
            cluster_skew = 0.5,
            seed = spec.seed,
        )
        train=generate_clustered_queries(
            spec.train_query_count,
            dataset.centers;
            query_noise = 0.08,
            cluster_skew = 0.5,
            seed = spec.seed+1,
        ).queries
        heldout=generate_clustered_queries(
            spec.heldout_query_count,
            dataset.centers;
            query_noise = 0.08,
            cluster_skew = 0.5,
            seed = spec.seed+2,
        ).queries
        vectors=dataset.vectors
    else
        dataset=generate_synthetic_dataset(
            spec.vector_count,
            spec.dimension;
            seed = spec.seed,
        )
        vectors=dataset.vectors
        train=generate_synthetic_queries(
            spec.train_query_count,
            spec.dimension;
            seed = spec.seed+1,
        )
        heldout=generate_synthetic_queries(
            spec.heldout_query_count,
            spec.dimension;
            seed = spec.seed+2,
        )
    end

    return (vectors = vectors, train_queries = train, heldout_queries = heldout)
end

function generate_experiment_data(spec::ExperimentSpec; base = nothing)
    resolved_base=base===nothing ? generate_experiment_base(spec) : base
    size(resolved_base.vectors)==(spec.dimension, spec.vector_count)||throw(
        DimensionMismatch("experiment base doesnt match spec"),
    )
    size(resolved_base.train_queries)==(spec.dimension, spec.train_query_count)||throw(
        DimensionMismatch("train queries dont match spec"),
    )
    size(resolved_base.heldout_queries)==(spec.dimension, spec.heldout_query_count)||throw(
        DimensionMismatch("heldout queries dont match spec"),
    )

    metadata=generate_filter_metadata(
        resolved_base.vectors,
        spec.selectivity;
        workload = spec.filter_workload,
        seed = spec.seed+3,
    )
    train_filters=fill((selected = true,), spec.train_query_count)
    heldout_filters=fill((selected = true,), spec.heldout_query_count)

    return (
        vectors = resolved_base.vectors,
        metadata = metadata,
        train_queries = resolved_base.train_queries,
        heldout_queries = resolved_base.heldout_queries,
        train_filters = train_filters,
        heldout_filters = heldout_filters,
    )
end

function benchmark_database(
    context::BenchmarkContext,
    vectors::AbstractMatrix,
    metadata::AbstractVector,
    metric::Symbol,
)
    dim, count=size(vectors)
    length(metadata)==count||throw(DimensionMismatch("metadata count doesnt match vectors"))
    context.filter_index.count==count||throw(
        DimensionMismatch("benchmark context doesnt match vectors"),
    )
    context.index.ivf.metric===metric||throw(
        ArgumentError("benchmark metric doesnt match index"),
    )

    stored_vectors=vectors isa Matrix{Float32} ? vectors : Matrix{Float32}(vectors)
    vector_store=VectorStore(dim, stored_vectors, count)
    metadata_store=MetadataStore(NamedTuple[row for row in metadata])
    ids=Any[index for index = 1:count]
    positions=Dict{Any,Int}(id=>index for (index, id) in enumerate(ids))
    id_store=IDStore(ids, positions)
    revision=UInt64(count)

    return VectorDB(
        "benchmark",
        dim,
        metric,
        vector_store,
        metadata_store,
        id_store,
        context.index,
        context.filter_index,
        nothing,
        revision,
        revision,
        DatabaseLock(),
        Dict{Any,Any}(),
        ReentrantLock(),
    )
end

function search_benchmark_method(
    context::BenchmarkContext,
    vectors::AbstractMatrix,
    metadata::AbstractVector,
    query::AbstractVector,
    filter::Union{NamedTuple,FilterExpr},
    method::Symbol,
    nprobe::Int,
    spec::ExperimentSpec;
    postfilter_candidate_multiplier::Float64 = spec.candidate_multiplier,
)
    if method===:exact
        return search_exact(
            vectors,
            metadata,
            query;
            k = spec.k,
            metric = spec.metric,
            filter = filter,
            filter_index = context.filter_index,
            vector_norms = context.index.ivf.vector_norms,
        )
    elseif method===:ivf_prefilter
        return search_ivf_prefilter(
            context.index,
            vectors,
            metadata,
            query;
            k = spec.k,
            nprobe = nprobe,
            metric = spec.metric,
            filter = filter,
        )
    elseif method===:ivf_postfilter
        selectivity=estimate_selectivity(context.filter_index, filter)
        oversample=resolve_postfilter_oversample(
            10,
            selectivity;
            candidate_multiplier = postfilter_candidate_multiplier,
        )
        return search_ivf_postfilter(
            context.index.ivf,
            vectors,
            metadata,
            query;
            k = spec.k,
            nprobe = nprobe,
            metric = spec.metric,
            filter = filter,
            oversample = oversample,
        )
    elseif method===:filter_aware
        return search_filter_aware_ivf(
            context.index,
            vectors,
            metadata,
            query;
            k = spec.k,
            nprobe = nprobe,
            metric = spec.metric,
            filter = filter,
            adaptive = false,
            vector_weight = spec.vector_weight,
            filter_weight = spec.filter_weight,
            rerank_factor = spec.rerank_factor,
        )
    elseif method===:filter_aware_bound
        return search_filter_aware_bound(
            context.index,
            vectors,
            metadata,
            query;
            k = spec.k,
            minimum_nprobe = 1,
            max_nprobe = nprobe,
            metric = spec.metric,
            filter = filter,
        )
    end

    throw(ArgumentError("unsupported benchmark method"))
end

function benchmark_method_work(
    context::BenchmarkContext,
    vectors::AbstractMatrix,
    metadata::AbstractVector,
    query::AbstractVector,
    filter::Union{NamedTuple,FilterExpr},
    method::Symbol,
    nprobe::Int,
    spec::ExperimentSpec,
)
    if method===:exact
        count=count_exact_candidates(context.filter_index, filter)
        return (visited = count, scored = count, probed = 0)
    elseif method===:ivf_prefilter
        work=count_ivf_prefilter_work(
            context.index,
            metadata,
            query,
            filter;
            nprobe = nprobe,
        )
        return (visited = work.visited, scored = work.scored, probed = work.probed_lists)
    elseif method===:ivf_postfilter
        count=count_ivf_candidates(context.index.ivf, query; nprobe = nprobe)
        return (visited = count, scored = count, probed = nprobe)
    elseif method===:filter_aware
        work=count_filter_aware_work(
            context.index,
            metadata,
            query,
            filter;
            k = spec.k,
            nprobe = nprobe,
            adaptive = false,
            vector_weight = spec.vector_weight,
            filter_weight = spec.filter_weight,
            rerank_factor = spec.rerank_factor,
        )
        return (visited = work.visited, scored = work.scored, probed = work.probed_lists)
    elseif method===:filter_aware_bound
        stats=search_filter_aware_bound_with_stats(
            context.index,
            vectors,
            metadata,
            query;
            k = spec.k,
            minimum_nprobe = 1,
            max_nprobe = nprobe,
            metric = spec.metric,
            filter = filter,
        )
        return (visited = stats.visited, scored = stats.scored, probed = stats.probed_lists)
    end

    throw(ArgumentError("unsupported benchmark method"))
end

function execute_benchmark_plan(
    context::BenchmarkContext,
    vectors::AbstractMatrix,
    metadata::AbstractVector,
    query::AbstractVector,
    filter::Union{NamedTuple,FilterExpr},
    plan::QueryPlan,
    spec::ExperimentSpec;
    postfilter_candidate_multiplier::Float64 = spec.candidate_multiplier,
)
    method=strategy_name(plan.strategy)

    if method===:exact
        return search_benchmark_method(
            context,
            vectors,
            metadata,
            query,
            filter,
            :exact,
            0,
            spec,
        )
    elseif method===:filter_aware
        minimum_nprobe=max(1, plan.minimum_nprobe)
        maximum_nprobe=max(minimum_nprobe, plan.nprobe)
        return search_filter_aware_ivf(
            context.index,
            vectors,
            metadata,
            query;
            k = spec.k,
            nprobe = minimum_nprobe,
            metric = spec.metric,
            filter = filter,
            adaptive = true,
            max_nprobe = maximum_nprobe,
            candidate_multiplier = spec.candidate_multiplier,
            vector_weight = spec.vector_weight,
            filter_weight = spec.filter_weight,
            rerank_factor = spec.rerank_factor,
        )
    elseif method===:filter_aware_bound
        minimum_nprobe=max(1, plan.minimum_nprobe)
        maximum_nprobe=max(minimum_nprobe, plan.nprobe)
        return search_filter_aware_bound(
            context.index,
            vectors,
            metadata,
            query;
            k = spec.k,
            minimum_nprobe = minimum_nprobe,
            max_nprobe = maximum_nprobe,
            metric = spec.metric,
            filter = filter,
        )
    end

    return search_benchmark_method(
        context,
        vectors,
        metadata,
        query,
        filter,
        method,
        max(1, plan.nprobe),
        spec;
        postfilter_candidate_multiplier = postfilter_candidate_multiplier,
    )
end

function benchmark_plan_work(
    context::BenchmarkContext,
    vectors::AbstractMatrix,
    metadata::AbstractVector,
    query::AbstractVector,
    filter::Union{NamedTuple,FilterExpr},
    plan::QueryPlan,
    spec::ExperimentSpec,
)
    method=strategy_name(plan.strategy)

    if method===:filter_aware
        minimum_nprobe=max(1, plan.minimum_nprobe)
        maximum_nprobe=max(minimum_nprobe, plan.nprobe)
        work=count_filter_aware_work(
            context.index,
            metadata,
            query,
            filter;
            k = spec.k,
            nprobe = minimum_nprobe,
            adaptive = true,
            max_nprobe = maximum_nprobe,
            candidate_multiplier = spec.candidate_multiplier,
            vector_weight = spec.vector_weight,
            filter_weight = spec.filter_weight,
            rerank_factor = spec.rerank_factor,
        )
        return (visited = work.visited, scored = work.scored, probed = work.probed_lists)
    elseif method===:filter_aware_bound
        minimum_nprobe=max(1, plan.minimum_nprobe)
        maximum_nprobe=max(minimum_nprobe, plan.nprobe)
        stats=search_filter_aware_bound_with_stats(
            context.index,
            vectors,
            metadata,
            query;
            k = spec.k,
            minimum_nprobe = minimum_nprobe,
            max_nprobe = maximum_nprobe,
            metric = spec.metric,
            filter = filter,
        )
        return (visited = stats.visited, scored = stats.scored, probed = stats.probed_lists)
    end

    return benchmark_method_work(
        context,
        vectors,
        metadata,
        query,
        filter,
        method,
        method===:exact ? 0 : max(1, plan.nprobe),
        spec,
    )
end

function evaluate_benchmark_method(
    spec::ExperimentSpec,
    context::BenchmarkContext,
    vectors::AbstractMatrix,
    metadata::AbstractVector,
    queries::AbstractMatrix,
    filters::AbstractVector,
    truth_ids::AbstractVector,
    method::Symbol,
    nprobe::Int;
    split::Symbol = :heldout,
    db::Union{Nothing,VectorDB} = nothing,
    planner_config::Union{Nothing,PlannerConfig} = nothing,
    postfilter_candidate_multiplier::Float64 = spec.candidate_multiplier,
)
    query_count=size(queries, 2)
    length(filters)==query_count||throw(
        DimensionMismatch("filter count doesnt match queries"),
    )
    length(truth_ids)==query_count||throw(
        DimensionMismatch("groundtruth count doesnt match queries"),
    )
    method===:auto&&(db===nothing||planner_config===nothing)&&throw(
        ArgumentError("auto benchmark requires a database and planner config"),
    )

    recalls=Float64[]
    result_counts=Int[]
    visited_counts=Int[]
    scored_counts=Int[]
    probe_counts=Int[]
    times_ns=Int[]
    rows=NamedTuple[]
    selected_methods=Symbol[]

    for query_index = 1:query_count
        query=@view queries[:, query_index]
        filter=filters[query_index]
        actual_method=method
        actual_nprobe=nprobe
        plan=nothing

        search_function=if method===:auto
            ()->begin
                selected_plan=cached_plan_query(
                    db,
                    filter;
                    k = spec.k,
                    config = planner_config,
                )
                results=execute_benchmark_plan(
                    context,
                    vectors,
                    metadata,
                    query,
                    filter,
                    selected_plan,
                    spec;
                    postfilter_candidate_multiplier = planner_config.postfilter_candidate_multiplier,
                )
                return (plan = selected_plan, results = results)
            end
        else
            ()->search_benchmark_method(
                context,
                vectors,
                metadata,
                query,
                filter,
                method,
                nprobe,
                spec;
                postfilter_candidate_multiplier = postfilter_candidate_multiplier,
            )
        end

        measurement=measure_latency(search_function; repetitions = spec.repetitions)
        results=if method===:auto
            plan=measurement.result.plan
            actual_method=strategy_name(plan.strategy)
            actual_nprobe=actual_method===:exact ? 0 : plan.nprobe
            measurement.result.results
        else
            measurement.result
        end

        work=method===:auto ?
             benchmark_plan_work(context, vectors, metadata, query, filter, plan, spec) :
             benchmark_method_work(
            context,
            vectors,
            metadata,
            query,
            filter,
            method,
            nprobe,
            spec,
        )
        predicted=Int[result.index for result in results]
        recall=recall_at_k(predicted, truth_ids[query_index], spec.k)

        push!(recalls, recall)
        push!(result_counts, length(results))
        push!(visited_counts, work.visited)
        push!(scored_counts, work.scored)
        push!(probe_counts, work.probed)
        push!(selected_methods, actual_method)
        append!(times_ns, measurement.times_ns)

        for (repetition, time_value) in enumerate(measurement.times_ns)
            resolved_postfilter_multiplier=method===:auto ?
                                           planner_config.postfilter_candidate_multiplier :
                                           postfilter_candidate_multiplier
            push!(
                rows,
                (
                    experiment = spec.name,
                    split = String(split),
                    vector_count = spec.vector_count,
                    dimension = spec.dimension,
                    vector_workload = String(spec.vector_workload),
                    filter_workload = String(spec.filter_workload),
                    selectivity = spec.selectivity,
                    seed = spec.seed,
                    method = String(method),
                    selected_method = String(actual_method),
                    query_index = query_index,
                    repetition = repetition,
                    nprobe = actual_nprobe,
                    recall = recall,
                    latency_ms = time_value/1_000_000,
                    candidates_visited = work.visited,
                    candidates_scored = work.scored,
                    lists_probed = work.probed,
                    result_count = length(results),
                    postfilter_candidate_multiplier = actual_method===:ivf_postfilter ?
                                                      resolved_postfilter_multiplier : 0.0,
                ),
            )
        end
    end

    summary=benchmark_summary(
        recalls,
        result_counts,
        visited_counts,
        scored_counts,
        probe_counts,
        times_ns,
    )
    return (
        method = method,
        nprobe = nprobe,
        summary = summary,
        rows = rows,
        selected_methods = selected_methods,
    )
end

function select_sweep_method(
    sweep_results::AbstractVector,
    method::Symbol,
    target_recall::Float64,
    selection_margin::Float64;
    probe_safety_factor::Float64 = 2.0,
    postfilter_candidate_multiplier::Float64 = 0.0,
)
    points=sweep_points(sweep_results, method)
    isempty(points)&&throw(ArgumentError("sweep method is missing"))
    probe_safety_factor>=1.0||throw(ArgumentError("probe safety factor must be atleast 1"))

    if method===:exact
        point=first(points)
        return (
            method = method,
            nprobe = 0,
            train_recall = point.recall,
            train_p50_ms = point.p50_ms,
            train_p95_ms = point.p95_ms,
            train_candidates = point.candidates_scored,
            achieved = point.recall>=target_recall,
            selection_target = target_recall,
            postfilter_candidate_multiplier = postfilter_candidate_multiplier,
        )
    end

    selection_target=min(1.0, target_recall+selection_margin)
    eligible=[point for point in points if point.recall>=selection_target]

    if isempty(eligible)&&selection_target>target_recall
        selection_target=target_recall
        eligible=[point for point in points if point.recall>=selection_target]
    end

    base_selection=if isempty(eligible)
        first(
            sort(
                points;
                by = point->(
                    -point.recall,
                    point.nprobe,
                    point.candidates_scored,
                    point.p95_ms,
                ),
            ),
        )
    else
        first(
            sort(
                eligible;
                by = point->(point.nprobe, point.candidates_scored, point.p95_ms),
            ),
        )
    end
    safety_nprobe=min(
        maximum(point.nprobe for point in points),
        ceil(Int, base_selection.nprobe*probe_safety_factor),
    )
    safety_points=[point for point in points if point.nprobe>=safety_nprobe]
    selected=first(
        sort(
            safety_points;
            by = point->(
                point.nprobe,
                -point.recall,
                point.candidates_scored,
                point.p95_ms,
            ),
        ),
    )

    return (
        method = method,
        nprobe = selected.nprobe,
        train_recall = selected.recall,
        train_p50_ms = selected.p50_ms,
        train_p95_ms = selected.p95_ms,
        train_candidates = selected.candidates_scored,
        achieved = selected.recall>=target_recall,
        selection_target = selection_target,
        postfilter_candidate_multiplier = postfilter_candidate_multiplier,
    )
end

function select_benchmark_methods(spec::ExperimentSpec, sweep_results::AbstractVector)
    return [
        select_sweep_method(
            sweep_results,
            method,
            spec.target_recall,
            spec.selection_margin;
            probe_safety_factor = spec.probe_safety_factor,
            postfilter_candidate_multiplier = method===:ivf_postfilter ?
                                              spec.candidate_multiplier : 0.0,
        ) for method in experiment_methods(spec)
    ]
end

function postfilter_multiplier_values(spec::ExperimentSpec)
    full_scan_multiplier=spec.selectivity==0 ? spec.candidate_multiplier :
                         spec.vector_count*spec.selectivity/spec.k
    multipliers=unique(
        vcat(
            spec.postfilter_multipliers,
            [spec.candidate_multiplier, full_scan_multiplier],
        ),
    )
    value=maximum(spec.postfilter_multipliers)

    while value<full_scan_multiplier
        value=min(value*2, full_scan_multiplier)
        push!(multipliers, value)
    end

    return sort!(unique(multipliers))
end

function build_postfilter_rankings(
    spec::ExperimentSpec,
    index::IVFIndex,
    vectors::AbstractMatrix,
    queries::AbstractMatrix,
)
    size(vectors)==(spec.dimension, spec.vector_count)||throw(
        DimensionMismatch("vectors dont match experiment spec"),
    )
    size(queries)==(spec.dimension, spec.train_query_count)||throw(
        DimensionMismatch("queries dont match experiment spec"),
    )
    length(index.lists)==spec.nlists||throw(
        DimensionMismatch("index doesnt match experiment spec"),
    )
    index.metric===spec.metric||throw(
        ArgumentError("index metric doesnt match experiment spec"),
    )
    maximum(spec.nprobes)<=length(index.lists)||throw(
        DimensionMismatch("nprobes exceed index list count"),
    )
    probe_set=Set(spec.nprobes)
    query_rankings=Vector{Dict{Int,Vector{Int}}}(undef, spec.train_query_count)

    for query_index = 1:spec.train_query_count
        query=@view queries[:, query_index]
        selected_lists=rank_ivf_lists(index, query; nprobe = maximum(spec.nprobes))
        candidate_indices=Int[]
        candidate_scores=Float32[]
        query_norm=spec.metric===:cosine ? vector_norm(query) : 1.0f0
        spec.metric===:cosine&&iszero(query_norm)&&throw(
            ArgumentError("query vector cannot be zero"),
        )
        rankings=Dict{Int,Vector{Int}}()

        for (probe_count, list_index) in enumerate(selected_lists)
            for vector_index in index.lists[list_index]
                score=column_dot(query, vectors, vector_index)

                if spec.metric===:cosine
                    stored_norm=isempty(index.vector_norms) ?
                                column_norm(vectors, vector_index) :
                                index.vector_norms[vector_index]
                    iszero(stored_norm)&&throw(
                        ArgumentError("stored vector cannot be zero"),
                    )
                    score/=query_norm*stored_norm
                end

                push!(candidate_indices, vector_index)
                push!(candidate_scores, score)
            end

            probe_count in probe_set||continue
            order=sortperm(candidate_scores; rev = true)
            rankings[probe_count]=candidate_indices[order]
        end

        query_rankings[query_index]=rankings
    end

    return (
        vector_count = spec.vector_count,
        dimension = spec.dimension,
        query_count = spec.train_query_count,
        nlists = spec.nlists,
        nprobes = copy(spec.nprobes),
        metric = spec.metric,
        rankings = query_rankings,
    )
end

function validate_postfilter_rankings(spec::ExperimentSpec, rankings)
    rankings.vector_count==spec.vector_count||throw(
        DimensionMismatch("postfilter rankings vector count doesnt match spec"),
    )
    rankings.dimension==spec.dimension||throw(
        DimensionMismatch("postfilter rankings dimension doesnt match spec"),
    )
    rankings.query_count==spec.train_query_count||throw(
        DimensionMismatch("postfilter rankings query count doesnt match spec"),
    )
    rankings.nlists==spec.nlists||throw(
        DimensionMismatch("postfilter rankings list count doesnt match spec"),
    )
    rankings.nprobes==spec.nprobes||throw(
        DimensionMismatch("postfilter rankings nprobes dont match spec"),
    )
    rankings.metric===spec.metric||throw(
        ArgumentError("postfilter rankings metric doesnt match spec"),
    )
    length(rankings.rankings)==spec.train_query_count||throw(
        DimensionMismatch("postfilter ranking rows dont match spec"),
    )
    return rankings
end

function postfilter_prefix_ids(
    ranked_ids::AbstractVector{<:Integer},
    metadata::AbstractVector,
    filter::Union{NamedTuple,FilterExpr},
    k::Int,
    cutoff::Int,
)
    ids=Int[]
    sizehint!(ids, k)

    for position = 1:min(cutoff, length(ranked_ids))
        vector_index=Int(ranked_ids[position])
        matches_filter(metadata[vector_index], filter)||continue
        push!(ids, vector_index)
        length(ids)==k&&break
    end

    return ids
end

function select_postfilter_configuration(
    spec::ExperimentSpec,
    context::BenchmarkContext,
    vectors::AbstractMatrix,
    metadata::AbstractVector,
    queries::AbstractMatrix,
    filters::AbstractVector,
    truth_ids::AbstractVector;
    prebuilt_rankings = nothing,
)
    query_count=size(queries, 2)
    length(filters)==query_count||throw(
        DimensionMismatch("filter count doesnt match queries"),
    )
    length(truth_ids)==query_count||throw(
        DimensionMismatch("groundtruth count doesnt match queries"),
    )
    ranking_cache=prebuilt_rankings===nothing ?
                  build_postfilter_rankings(spec, context.index.ivf, vectors, queries) :
                  validate_postfilter_rankings(spec, prebuilt_rankings)
    multipliers=postfilter_multiplier_values(spec)
    points=NamedTuple[]

    for nprobe in spec.nprobes
        recall_totals=zeros(Float64, length(multipliers))
        cutoff_totals=zeros(Float64, length(multipliers))
        candidate_total=0.0

        for query_index = 1:query_count
            filter=filters[query_index]
            ranked=ranking_cache.rankings[query_index][nprobe]
            candidate_count=length(ranked)
            candidate_total+=candidate_count

            for (multiplier_index, multiplier) in enumerate(multipliers)
                oversample=resolve_postfilter_oversample(
                    10,
                    spec.selectivity;
                    candidate_multiplier = multiplier,
                )
                cutoff=min(length(ranked), spec.k*oversample)
                predicted=postfilter_prefix_ids(ranked, metadata, filter, spec.k, cutoff)
                recall_totals[multiplier_index]+=recall_at_k(
                    predicted,
                    truth_ids[query_index],
                    spec.k,
                )
                cutoff_totals[multiplier_index]+=cutoff
            end
        end

        for (multiplier_index, multiplier) in enumerate(multipliers)
            push!(
                points,
                (
                    nprobe = nprobe,
                    postfilter_candidate_multiplier = multiplier,
                    recall = recall_totals[multiplier_index]/query_count,
                    candidates_scored = candidate_total/query_count,
                    ranked_cutoff = cutoff_totals[multiplier_index]/query_count,
                ),
            )
        end
    end

    selection_target=min(1.0, spec.target_recall+spec.selection_margin)
    eligible=[point for point in points if point.recall>=selection_target]

    if isempty(eligible)&&selection_target>spec.target_recall
        selection_target=spec.target_recall
        eligible=[point for point in points if point.recall>=selection_target]
    end

    base_selection=if isempty(eligible)
        first(
            sort(
                points;
                by = point->(
                    -point.recall,
                    point.candidates_scored,
                    point.ranked_cutoff,
                    point.nprobe,
                    point.postfilter_candidate_multiplier,
                ),
            ),
        )
    else
        first(
            sort(
                eligible;
                by = point->(
                    point.candidates_scored,
                    point.ranked_cutoff,
                    point.nprobe,
                    point.postfilter_candidate_multiplier,
                ),
            ),
        )
    end
    safety_nprobe=min(
        maximum(point.nprobe for point in points),
        ceil(Int, base_selection.nprobe*spec.probe_safety_factor),
    )
    safety_multiplier=base_selection.postfilter_candidate_multiplier*spec.postfilter_safety_factor
    safety_points=[
        point for point in points if
        point.nprobe>=safety_nprobe&&point.postfilter_candidate_multiplier>=safety_multiplier&&(
            isempty(eligible)||point.recall>=selection_target
        )
    ]

    if isempty(safety_points)
        safety_points=[
            point for point in points if
            point.nprobe>=safety_nprobe&&point.postfilter_candidate_multiplier>=safety_multiplier
        ]
    end

    if isempty(safety_points)
        safety_points=[point for point in points if point.nprobe>=safety_nprobe]
    end

    selected=isempty(eligible) ?
             first(
        sort(
            safety_points;
            by = point->(
                -point.recall,
                point.candidates_scored,
                point.ranked_cutoff,
                point.nprobe,
                point.postfilter_candidate_multiplier,
            ),
        ),
    ) :
             first(
        sort(
            safety_points;
            by = point->(
                point.candidates_scored,
                point.ranked_cutoff,
                point.nprobe,
                point.postfilter_candidate_multiplier,
            ),
        ),
    )

    evaluation=evaluate_benchmark_method(
        spec,
        context,
        vectors,
        metadata,
        queries,
        filters,
        truth_ids,
        :ivf_postfilter,
        selected.nprobe;
        split = :train,
        postfilter_candidate_multiplier = selected.postfilter_candidate_multiplier,
    )

    return (
        method = :ivf_postfilter,
        nprobe = selected.nprobe,
        train_recall = evaluation.summary.average_recall,
        train_p50_ms = evaluation.summary.latency.p50_ms,
        train_p95_ms = evaluation.summary.latency.p95_ms,
        train_candidates = evaluation.summary.average_candidates_scored,
        achieved = evaluation.summary.average_recall>=spec.target_recall,
        selection_target = selection_target,
        postfilter_candidate_multiplier = selected.postfilter_candidate_multiplier,
    )
end

function run_claim_benchmark(
    spec::ExperimentSpec;
    base = nothing,
    prebuilt_ivf::Union{Nothing,IVFIndex} = nothing,
    prebuilt_build_seconds::Union{Nothing,Float64} = nothing,
    prebuilt_postfilter_rankings = nothing,
    prebuilt_postfilter_ranking_seconds::Union{Nothing,Float64} = nothing,
)
    data=generate_experiment_data(spec; base = base)
    context=nothing
    build_seconds=0.0

    if prebuilt_ivf===nothing
        build_seconds=@elapsed context=build_benchmark_context(
            data.vectors,
            data.metadata;
            nlists = spec.nlists,
            iterations = spec.iterations,
            seed = spec.seed,
            metric = spec.metric,
            restarts = spec.restarts,
            training_count = spec.training_count,
        )
    else
        prebuilt_ivf.metric===spec.metric||throw(
            ArgumentError("prebuilt index metric doesnt match spec"),
        )
        length(prebuilt_ivf.lists)==spec.nlists||throw(
            DimensionMismatch("prebuilt list count doesnt match spec"),
        )
        sum(length, prebuilt_ivf.lists)==spec.vector_count||throw(
            DimensionMismatch("prebuilt index count doesnt match spec"),
        )
        context=build_benchmark_context(prebuilt_ivf, data.metadata)
        build_seconds=prebuilt_build_seconds===nothing ? 0.0 : prebuilt_build_seconds
    end

    train_sweep=run_nprobe_sweep(
        context,
        data.vectors,
        data.metadata,
        data.train_queries,
        data.train_filters,
        spec.nprobes;
        k = spec.k,
        repetitions = spec.repetitions,
        metric = spec.metric,
        candidate_multiplier = spec.candidate_multiplier,
        vector_weight = spec.vector_weight,
        filter_weight = spec.filter_weight,
        rerank_factor = spec.rerank_factor,
    )
    train_truth_ids=compute_groundtruth(
        data.vectors,
        data.metadata,
        data.train_queries;
        k = spec.k,
        metric = spec.metric,
        filters = data.train_filters,
    )
    selections=select_benchmark_methods(spec, train_sweep)
    postfilter_rankings=prebuilt_postfilter_rankings
    postfilter_ranking_seconds=prebuilt_postfilter_ranking_seconds===nothing ? 0.0 :
                               prebuilt_postfilter_ranking_seconds

    if postfilter_rankings===nothing
        postfilter_ranking_seconds=@elapsed postfilter_rankings=build_postfilter_rankings(
            spec,
            context.index.ivf,
            data.vectors,
            data.train_queries,
        )
    end

    postfilter_selection=select_postfilter_configuration(
        spec,
        context,
        data.vectors,
        data.metadata,
        data.train_queries,
        data.train_filters,
        train_truth_ids;
        prebuilt_rankings = postfilter_rankings,
    )
    selections=[
        selection.method===:ivf_postfilter ? postfilter_selection : selection for
        selection in selections
    ]
    calibration=RecallCalibration()

    for selection in selections
        selection.method===:exact&&continue
        add_calibration_entry!(
            calibration,
            RecallCalibrationEntry(
                selection.method,
                spec.filter_workload,
                spec.selectivity,
                spec.vector_count,
                spec.nlists,
                spec.target_recall,
                selection.nprobe,
                selection.train_recall,
                selection.train_p50_ms,
                selection.achieved,
                spec.dimension,
                spec.metric,
                1,
            ),
        )
    end
    planner_config=PlannerConfig(
        target_recall = spec.target_recall,
        candidate_multiplier = spec.candidate_multiplier,
        postfilter_candidate_multiplier = postfilter_selection.postfilter_candidate_multiplier,
        default_nprobe = min(4, spec.nlists),
        max_nprobe = maximum(spec.nprobes),
        calibration = calibration,
        workload = spec.filter_workload,
    )
    db=benchmark_database(context, data.vectors, data.metadata, spec.metric)
    truth_ids=compute_groundtruth(
        data.vectors,
        data.metadata,
        data.heldout_queries;
        k = spec.k,
        metric = spec.metric,
        filters = data.heldout_filters,
    )
    method_results=Dict{Symbol,Any}()
    rows=NamedTuple[]

    for selection in selections
        result=evaluate_benchmark_method(
            spec,
            context,
            data.vectors,
            data.metadata,
            data.heldout_queries,
            data.heldout_filters,
            truth_ids,
            selection.method,
            selection.nprobe;
            postfilter_candidate_multiplier = selection.postfilter_candidate_multiplier,
        )
        method_results[selection.method]=result
        append!(rows, result.rows)
    end

    auto_result=evaluate_benchmark_method(
        spec,
        context,
        data.vectors,
        data.metadata,
        data.heldout_queries,
        data.heldout_filters,
        truth_ids,
        :auto,
        0;
        db = db,
        planner_config = planner_config,
    )
    method_results[:auto]=auto_result
    append!(rows, auto_result.rows)
    selection_map=Dict(selection.method=>selection for selection in selections)

    eligible=NamedTuple[]
    for selection in selections
        result=method_results[selection.method]
        selection.achieved&&result.summary.average_recall>=spec.target_recall||continue
        push!(eligible, (method = selection.method, result = result))
    end
    oracle=isempty(eligible) ? nothing :
           first(sort(eligible; by = value->value.result.summary.latency.p95_ms))
    planner_regret=oracle===nothing ? Inf :
                   auto_result.summary.latency.p95_ms/oracle.result.summary.latency.p95_ms

    postfilter=get(method_results, :ivf_postfilter, nothing)
    postfilter_selection=get(selection_map, :ivf_postfilter, nothing)
    aware_candidates=NamedTuple[]

    for method in (:filter_aware, :filter_aware_bound)
        haskey(method_results, method)||continue
        selection=selection_map[method]
        result=method_results[method]
        selection.achieved&&result.summary.average_recall>=spec.target_recall||continue
        push!(aware_candidates, (method = method, result = result))
    end

    valid_postfilter=postfilter!==nothing&&postfilter_selection!==nothing&&postfilter_selection.achieved&&postfilter.summary.average_recall>=spec.target_recall
    comparable_aware=valid_postfilter ?
                     [
        value for value in aware_candidates if
        value.result.summary.average_recall+spec.recall_tolerance>=postfilter.summary.average_recall
    ] : NamedTuple[]
    best_aware=isempty(comparable_aware) ? nothing :
               first(
        sort(comparable_aware; by = value->value.result.summary.latency.p95_ms),
    )
    comparable=valid_postfilter&&best_aware!==nothing
    speedup=comparable ?
            postfilter.summary.latency.p95_ms/best_aware.result.summary.latency.p95_ms : 0.0
    claim=(
        passed = comparable&&speedup>=spec.minimum_speedup,
        comparable = comparable,
        aware_method = best_aware===nothing ? :none : best_aware.method,
        speedup_p95 = speedup,
        postfilter_recall = postfilter===nothing ? 0.0 : postfilter.summary.average_recall,
        aware_recall = best_aware===nothing ? 0.0 :
                       best_aware.result.summary.average_recall,
    )

    return (
        spec = spec,
        build_seconds = build_seconds,
        index_megabytes = Base.summarysize(context)/1024^2,
        postfilter_ranking_seconds = postfilter_ranking_seconds,
        postfilter_ranking_megabytes = Base.summarysize(postfilter_rankings)/1024^2,
        maxrss_megabytes = Sys.maxrss()/1024^2,
        selections = selections,
        method_results = method_results,
        raw_rows = rows,
        postfilter_candidate_multiplier = postfilter_selection.postfilter_candidate_multiplier,
        oracle_method = oracle===nothing ? :none : oracle.method,
        planner_regret = planner_regret,
        claim = claim,
    )
end

function experiment_summary_rows(result)
    spec=result.spec
    selection_map=Dict(selection.method=>selection for selection in result.selections)
    rows=NamedTuple[]

    for method in vcat(experiment_methods(spec), [:auto])
        evaluation=result.method_results[method]
        selection=get(selection_map, method, nothing)
        postfilter_multiplier=method===:ivf_postfilter ?
                              selection.postfilter_candidate_multiplier :
                              method===:auto ? result.postfilter_candidate_multiplier : 0.0
        push!(
            rows,
            (
                experiment = spec.name,
                vector_count = spec.vector_count,
                dimension = spec.dimension,
                vector_workload = String(spec.vector_workload),
                filter_workload = String(spec.filter_workload),
                selectivity = spec.selectivity,
                seed = spec.seed,
                method = String(method),
                executed_methods = join(
                    sort!(unique(String.(evaluation.selected_methods))),
                    ',',
                ),
                selected_nprobe = selection===nothing ? 0 : selection.nprobe,
                selection_achieved = selection===nothing ? true : selection.achieved,
                train_recall = selection===nothing ? NaN : selection.train_recall,
                train_p50_ms = selection===nothing ? NaN : selection.train_p50_ms,
                train_p95_ms = selection===nothing ? NaN : selection.train_p95_ms,
                heldout_recall = evaluation.summary.average_recall,
                heldout_p50_ms = evaluation.summary.latency.p50_ms,
                heldout_p95_ms = evaluation.summary.latency.p95_ms,
                candidates_visited = evaluation.summary.average_candidates_visited,
                candidates_scored = evaluation.summary.average_candidates_scored,
                lists_probed = evaluation.summary.average_lists_probed,
                build_seconds = result.build_seconds,
                index_megabytes = result.index_megabytes,
                postfilter_ranking_seconds = result.postfilter_ranking_seconds,
                postfilter_ranking_megabytes = result.postfilter_ranking_megabytes,
                maxrss_megabytes = result.maxrss_megabytes,
                oracle_method = String(result.oracle_method),
                planner_regret = result.planner_regret,
                claim_passed = result.claim.passed,
                claim_speedup_p95 = result.claim.speedup_p95,
                probe_safety_factor = spec.probe_safety_factor,
                candidate_multiplier = spec.candidate_multiplier,
                postfilter_candidate_multiplier = postfilter_multiplier,
                postfilter_safety_factor = spec.postfilter_safety_factor,
                vector_weight = spec.vector_weight,
                filter_weight = spec.filter_weight,
                rerank_factor = spec.rerank_factor,
            ),
        )
    end

    return rows
end

function mean_number(values::AbstractVector{<:Real})
    isempty(values)&&throw(ArgumentError("values cannot be empty"))
    return sum(values)/length(values)
end

function median_number(values::AbstractVector{<:Real})
    isempty(values)&&throw(ArgumentError("values cannot be empty"))
    sorted=sort(Float64.(values))
    count=length(sorted)
    middle=count÷2
    return isodd(count) ? sorted[middle+1] : (sorted[middle]+sorted[middle+1])/2
end

function experiment_group_key(spec::ExperimentSpec)
    return (
        spec.vector_count,
        spec.dimension,
        spec.vector_workload,
        spec.filter_workload,
        spec.selectivity,
        spec.train_query_count,
        spec.heldout_query_count,
        spec.nlists,
        Tuple(spec.nprobes),
        spec.k,
        spec.target_recall,
        spec.selection_margin,
        spec.probe_safety_factor,
        spec.minimum_speedup,
        spec.recall_tolerance,
        spec.candidate_multiplier,
        Tuple(spec.postfilter_multipliers),
        spec.postfilter_safety_factor,
        spec.vector_weight,
        spec.filter_weight,
        spec.rerank_factor,
        spec.repetitions,
        spec.iterations,
        spec.restarts,
        spec.training_count,
        spec.metric,
    )
end

function group_experiment_results(results::AbstractVector)
    groups=Dict{Any,Vector{Any}}()

    for result in results
        push!(get!(groups, experiment_group_key(result.spec), Any[]), result)
    end

    return groups
end

function experiment_aggregate_rows(results::AbstractVector)
    isempty(results)&&throw(ArgumentError("experiment results cannot be empty"))
    rows=NamedTuple[]

    for group in values(group_experiment_results(results))
        first_result=first(group)
        spec=first_result.spec
        seeds=sort([result.spec.seed for result in group])
        regrets=[result.planner_regret for result in group]

        for method in vcat(experiment_methods(spec), [:auto])
            evaluations=[result.method_results[method] for result in group]
            recalls=[evaluation.summary.average_recall for evaluation in evaluations]
            p50_values=[evaluation.summary.latency.p50_ms for evaluation in evaluations]
            p95_values=[evaluation.summary.latency.p95_ms for evaluation in evaluations]
            scored=[
                evaluation.summary.average_candidates_scored for evaluation in evaluations
            ]
            probed=[evaluation.summary.average_lists_probed for evaluation in evaluations]
            postfilter_multipliers=method in (:ivf_postfilter, :auto) ?
                                   [
                result.postfilter_candidate_multiplier for result in group
            ] : fill(0.0, length(group))
            push!(
                rows,
                (
                    vector_count = spec.vector_count,
                    dimension = spec.dimension,
                    vector_workload = String(spec.vector_workload),
                    filter_workload = String(spec.filter_workload),
                    selectivity = spec.selectivity,
                    nlists = spec.nlists,
                    k = spec.k,
                    target_recall = spec.target_recall,
                    method = String(method),
                    runs = length(group),
                    seeds = join(seeds, ','),
                    recall_min = minimum(recalls),
                    recall_mean = mean_number(recalls),
                    recall_max = maximum(recalls),
                    p50_min_ms = minimum(p50_values),
                    p50_median_ms = median_number(p50_values),
                    p50_max_ms = maximum(p50_values),
                    p95_min_ms = minimum(p95_values),
                    p95_median_ms = median_number(p95_values),
                    p95_max_ms = maximum(p95_values),
                    candidates_scored_mean = mean_number(scored),
                    lists_probed_mean = mean_number(probed),
                    oracle_wins = count(result->result.oracle_method===method, group),
                    planner_regret_median = median_number(regrets),
                    planner_regret_max = maximum(regrets),
                    claim_passed = all(result->result.claim.passed, group),
                    probe_safety_factor = spec.probe_safety_factor,
                    candidate_multiplier = spec.candidate_multiplier,
                    postfilter_candidate_multiplier_median = median_number(
                        postfilter_multipliers,
                    ),
                    postfilter_safety_factor = spec.postfilter_safety_factor,
                    vector_weight = spec.vector_weight,
                    filter_weight = spec.filter_weight,
                    rerank_factor = spec.rerank_factor,
                ),
            )
        end
    end

    sort!(
        rows;
        by = row->(
            row.vector_count,
            row.dimension,
            row.vector_workload,
            row.filter_workload,
            row.selectivity,
            row.method,
        ),
    )
    return rows
end

function claim_aggregate_rows(results::AbstractVector)
    isempty(results)&&throw(ArgumentError("experiment results cannot be empty"))
    rows=NamedTuple[]

    for group in values(group_experiment_results(results))
        spec=first(group).spec
        speeds=[result.claim.speedup_p95 for result in group if result.claim.comparable]
        regrets=[result.planner_regret for result in group]
        seeds=sort([result.spec.seed for result in group])
        postfilter_multipliers=[result.postfilter_candidate_multiplier for result in group]
        push!(
            rows,
            (
                vector_count = spec.vector_count,
                dimension = spec.dimension,
                vector_workload = String(spec.vector_workload),
                filter_workload = String(spec.filter_workload),
                selectivity = spec.selectivity,
                target_recall = spec.target_recall,
                runs = length(group),
                seeds = join(seeds, ','),
                comparable_runs = length(speeds),
                passed_runs = count(result->result.claim.passed, group),
                all_passed = all(result->result.claim.passed, group),
                speedup_p95_min = isempty(speeds) ? 0.0 : minimum(speeds),
                speedup_p95_median = isempty(speeds) ? 0.0 : median_number(speeds),
                speedup_p95_max = isempty(speeds) ? 0.0 : maximum(speeds),
                planner_regret_median = median_number(regrets),
                planner_regret_max = maximum(regrets),
                aware_methods = join(
                    sort!(unique(String(result.claim.aware_method) for result in group)),
                    ',',
                ),
                probe_safety_factor = spec.probe_safety_factor,
                candidate_multiplier = spec.candidate_multiplier,
                postfilter_candidate_multiplier_min = minimum(postfilter_multipliers),
                postfilter_candidate_multiplier_median = median_number(
                    postfilter_multipliers,
                ),
                postfilter_candidate_multiplier_max = maximum(postfilter_multipliers),
                postfilter_safety_factor = spec.postfilter_safety_factor,
                vector_weight = spec.vector_weight,
                filter_weight = spec.filter_weight,
                rerank_factor = spec.rerank_factor,
            ),
        )
    end

    sort!(
        rows;
        by = row->(
            row.vector_count,
            row.dimension,
            row.vector_workload,
            row.filter_workload,
            row.selectivity,
        ),
    )
    return rows
end

function tsv_value(value)
    return replace(string(value), '\t'=>" ", '\n'=>" ", '\r'=>" ")
end

function write_tsv(path::AbstractString, rows::AbstractVector)
    isempty(rows)&&throw(ArgumentError("tsv rows cannot be empty"))
    mkpath(dirname(path))
    columns=propertynames(first(rows))
    all(row->propertynames(row)==columns, rows)||throw(
        ArgumentError("tsv rows must have the same columns"),
    )

    open(path, "w") do io
        println(io, join(String.(columns), '\t'))

        for row in rows
            println(
                io,
                join((tsv_value(getproperty(row, column)) for column in columns), '\t'),
            )
        end
    end

    return String(path)
end

function save_experiment_suite(path::AbstractString, results::AbstractVector)
    isempty(results)&&throw(ArgumentError("experiment results cannot be empty"))
    mkpath(path)
    raw_rows=NamedTuple[]
    summary_rows=NamedTuple[]

    for result in results
        append!(raw_rows, result.raw_rows)
        append!(summary_rows, experiment_summary_rows(result))
    end

    raw_path=write_tsv(joinpath(path, "raw_results.tsv"), raw_rows)
    summary_path=write_tsv(joinpath(path, "summary.tsv"), summary_rows)
    aggregate_path=write_tsv(
        joinpath(path, "aggregate.tsv"),
        experiment_aggregate_rows(results),
    )
    claims_path=write_tsv(joinpath(path, "claims.tsv"), claim_aggregate_rows(results))
    environment_path=joinpath(path, "environment.toml")
    environment=Dict(
        "julia_version"=>string(VERSION),
        "julia_threads"=>Threads.nthreads(),
        "kernel"=>string(Sys.KERNEL),
        "architecture"=>string(Sys.ARCH),
        "word_size"=>Sys.WORD_SIZE,
        "cpu_threads"=>Sys.CPU_THREADS,
        "generated_unix_time"=>time(),
        "experiments"=>[
            Dict(
                "name"=>result.spec.name,
                "vector_count"=>result.spec.vector_count,
                "dimension"=>result.spec.dimension,
                "vector_workload"=>String(result.spec.vector_workload),
                "filter_workload"=>String(result.spec.filter_workload),
                "selectivity"=>result.spec.selectivity,
                "seed"=>result.spec.seed,
                "target_recall"=>result.spec.target_recall,
                "selection_margin"=>result.spec.selection_margin,
                "probe_safety_factor"=>result.spec.probe_safety_factor,
                "minimum_speedup"=>result.spec.minimum_speedup,
                "recall_tolerance"=>result.spec.recall_tolerance,
                "train_query_count"=>result.spec.train_query_count,
                "heldout_query_count"=>result.spec.heldout_query_count,
                "nlists"=>result.spec.nlists,
                "nprobes"=>result.spec.nprobes,
                "k"=>result.spec.k,
                "repetitions"=>result.spec.repetitions,
                "iterations"=>result.spec.iterations,
                "restarts"=>result.spec.restarts,
                "training_count"=>result.spec.training_count,
                "metric"=>String(result.spec.metric),
                "candidate_multiplier"=>result.spec.candidate_multiplier,
                "postfilter_multipliers"=>result.spec.postfilter_multipliers,
                "postfilter_safety_factor"=>result.spec.postfilter_safety_factor,
                "selected_postfilter_candidate_multiplier"=>result.postfilter_candidate_multiplier,
                "postfilter_ranking_seconds"=>result.postfilter_ranking_seconds,
                "postfilter_ranking_megabytes"=>result.postfilter_ranking_megabytes,
                "vector_weight"=>result.spec.vector_weight,
                "filter_weight"=>result.spec.filter_weight,
                "rerank_factor"=>result.spec.rerank_factor,
            ) for result in results
        ],
    )

    open(environment_path, "w") do io
        TOML.print(io, environment)
    end

    return (
        raw = raw_path,
        summary = summary_path,
        aggregate = aggregate_path,
        claims = claims_path,
        environment = environment_path,
    )
end
