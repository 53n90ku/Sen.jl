using Random

function generate_selectivity_order(count::Int; seed::Int = 42)
    count>0||throw(ArgumentError("count must be positive"))
    return randperm(MersenneTwister(seed), count)
end

function generate_selectivity_metadata(
    order::AbstractVector{<:Integer},
    selectivity::Float64,
)
    count=length(order)
    count>0||throw(ArgumentError("order cannot be empty"))
    0.0<=selectivity<=1.0||throw(ArgumentError("selectivity must be between 0 and 1"))
    sort(Int.(order))==collect(1:count)||throw(
        ArgumentError("order must contain every index once"),
    )

    matching_count=round(Int, count*selectivity)
    selected=falses(count)
    selected[order[1:matching_count]].=true

    return [(selected = selected[index],) for index = 1:count]
end

function generate_selectivity_metadata(count::Int, selectivity::Float64; seed::Int = 42)
    order=generate_selectivity_order(count; seed = seed)
    return generate_selectivity_metadata(order, selectivity)
end

function run_selectivity_sweep(
    vectors::AbstractMatrix,
    queries::AbstractMatrix,
    selectivities::AbstractVector{<:Real};
    nlists::Int,
    k::Int = 10,
    nprobe::Int = 1,
    iterations::Int = 20,
    seed::Int = 42,
    repetitions::Int = 3,
    metric::Symbol = :cosine,
    postfilter_oversample::Int = 10,
    adaptive_postfilter::Bool = true,
    vector_weight::Float64 = 0.5,
    filter_weight::Float64 = 0.5,
    adaptive::Bool = false,
    max_nprobe::Int = nprobe,
    candidate_multiplier::Float64 = 4.0,
    rerank_factor::Int = 4,
)
    isempty(selectivities)&&throw(ArgumentError("selectivities cannot be empty"))

    _, vector_count=size(vectors)
    _, query_count=size(queries)
    query_count>0||throw(ArgumentError("queries cannot be empty"))

    order=generate_selectivity_order(vector_count; seed = seed)
    ivf=build_ivf(
        vectors;
        nlists = nlists,
        iterations = iterations,
        seed = seed,
        metric = metric,
    )
    sweep_results=NamedTuple[]

    for target_selectivity in selectivities
        selectivity=Float64(target_selectivity)
        metadata=generate_selectivity_metadata(order, selectivity)
        filters=fill((selected = true,), query_count)
        context=build_benchmark_context(ivf, metadata)

        benchmark=run_benchmark(
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
            vector_weight = vector_weight,
            filter_weight = filter_weight,
            adaptive = adaptive,
            max_nprobe = max_nprobe,
            candidate_multiplier = candidate_multiplier,
            rerank_factor = rerank_factor,
        )

        push!(
            sweep_results,
            (
                target_selectivity = selectivity,
                actual_selectivity = benchmark.average_selectivity,
                benchmark = benchmark,
            ),
        )
    end

    return sweep_results
end
