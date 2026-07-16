struct FilterAwareIVFIndex
    ivf::IVFIndex
    metadata_indexes::Vector{MetadataIndex}
end

function build_filter_aware_ivf(
    ivf::IVFIndex,
    metadata::AbstractVector,
)::FilterAwareIVFIndex
    vector_count=sum(length, ivf.lists)
    length(metadata)==vector_count||throw(
        DimensionMismatch("metadata count doesnt match vectors"),
    )

    metadata_indexes=Vector{MetadataIndex}(undef, length(ivf.lists))

    for list_index in eachindex(ivf.lists)
        metadata_indexes[list_index]=build_metadata_index(
            view(metadata, ivf.lists[list_index]),
        )
    end

    return FilterAwareIVFIndex(ivf, metadata_indexes)
end

function build_filter_aware_ivf(
    vectors::AbstractMatrix,
    metadata::AbstractVector;
    nlists::Int,
    iterations::Int = 20,
    seed::Int = 42,
    metric::Symbol = :cosine,
    restarts::Int = 1,
    training_count::Int = size(vectors, 2),
)::FilterAwareIVFIndex
    ivf=build_ivf(
        vectors;
        nlists = nlists,
        iterations = iterations,
        seed = seed,
        metric = metric,
        restarts = restarts,
        training_count = training_count,
    )
    return build_filter_aware_ivf(ivf, metadata)
end

function validate_list_filter_capability(metadata_index::MetadataIndex, filter::FilterExpr)
    if filter isa And||filter isa Or
        foreach(
            child->validate_list_filter_capability(metadata_index, child),
            filter.children,
        )
    elseif filter isa Not
        validate_list_filter_capability(metadata_index, filter.child)
    elseif !supports_indexed_filter(metadata_index, filter)
        field=filter isa Union{Eq,In,Range} ? " for field $(repr(filter.field))" : ""
        throw(
            ArgumentError(
                "list metadata index cannot evaluate $(nameof(typeof(filter)))$(field)",
            ),
        )
    end

    return filter
end

function _evaluate_list_filter(
    index::FilterAwareIVFIndex,
    list_index::Int,
    filter::FilterExpr,
)
    metadata_index=index.metadata_indexes[list_index]
    validate_list_filter_capability(metadata_index, filter)
    return evaluate_filter(metadata_index, filter)
end

function estimate_list_filter_density(
    index::FilterAwareIVFIndex,
    list_index::Int,
    filter::Union{NamedTuple,FilterExpr},
)::Float64
    1<=list_index<=length(index.ivf.lists)||throw(BoundsError(index.ivf.lists, list_index))
    normalized=normalize_filter(filter)

    list_size=length(index.ivf.lists[list_index])
    list_size==0&&return 0.0

    return count(_evaluate_list_filter(index, list_index, normalized))/list_size
end

function evaluate_list_filter(
    index::FilterAwareIVFIndex,
    list_index::Int,
    filter::Union{NamedTuple,FilterExpr},
)
    1<=list_index<=length(index.ivf.lists)||throw(BoundsError(index.ivf.lists, list_index))
    return _evaluate_list_filter(index, list_index, normalize_filter(filter))
end

function estimate_list_filter_count(
    index::FilterAwareIVFIndex,
    list_index::Int,
    filter::Union{NamedTuple,FilterExpr},
)
    return count(evaluate_list_filter(index, list_index, filter))
end

function filtered_list_candidates(
    index::FilterAwareIVFIndex,
    list_index::Int,
    filter::Union{NamedTuple,FilterExpr},
)
    1<=list_index<=length(index.ivf.lists)||throw(BoundsError(index.ivf.lists, list_index))
    mask=_evaluate_list_filter(index, list_index, normalize_filter(filter))
    list=index.ivf.lists[list_index]
    return Int[list[position] for position in eachindex(mask) if mask[position]]
end

function collect_filtered_list_candidates(
    index::FilterAwareIVFIndex,
    selected_lists::AbstractVector{<:Integer},
    filter::Union{NamedTuple,FilterExpr},
)
    return collect_filtered_list_candidates!(
        Int[],
        index,
        selected_lists,
        normalize_filter(filter),
    )
end

function collect_filtered_list_candidates!(
    candidate_indices::Vector{Int},
    index::FilterAwareIVFIndex,
    selected_lists::AbstractVector{<:Integer},
    filter::FilterExpr,
)
    empty!(candidate_indices)

    for list_index in selected_lists
        1<=list_index<=length(index.ivf.lists)||throw(
            BoundsError(index.ivf.lists, list_index),
        )
        mask=_evaluate_list_filter(index, list_index, filter)
        list=index.ivf.lists[list_index]

        for position in eachindex(mask)
            mask[position]&&push!(candidate_indices, list[position])
        end
    end

    return candidate_indices
end

function search_ivf_prefilter(
    index::FilterAwareIVFIndex,
    vectors::AbstractMatrix,
    metadata::AbstractVector,
    query::AbstractVector;
    k::Int = 10,
    nprobe::Int = 1,
    metric::Symbol = :cosine,
    filter::Union{NamedTuple,FilterExpr},
    excluded::Union{Nothing,BitVector} = nothing,
)
    validate_ivf_search(index.ivf, vectors, metadata, query)
    workspace=search_workspace()
    selected_lists=rank_ivf_lists!(
        workspace.selected_lists,
        index.ivf,
        query,
        nprobe,
        workspace,
    )
    candidate_indices=collect_filtered_list_candidates!(
        workspace.candidate_indices,
        index,
        selected_lists,
        normalize_filter(filter),
    )

    return score_ivf_candidates(
        vectors,
        metadata,
        query,
        candidate_indices;
        k = k,
        metric = metric,
        vector_norms = index.ivf.vector_norms,
        excluded = excluded,
        workspace = workspace,
    )
end

function normalized_scores(values::AbstractVector{<:Real})
    isempty(values)&&return Float64[]

    minimum_value=minimum(values)
    maximum_value=maximum(values)

    if minimum_value==maximum_value
        return zeros(Float64, length(values))
    end

    scale=maximum_value-minimum_value
    return [(Float64(value)-minimum_value)/scale for value in values]
end

function rank_filter_aware_lists(
    index::FilterAwareIVFIndex,
    query::AbstractVector,
    filter::Union{NamedTuple,FilterExpr};
    nprobe::Int,
    vector_weight::Float64 = 0.5,
    filter_weight::Float64 = 0.5,
    rerank_factor::Int = 4,
)::Vector{Int}
    dim, list_count=size(index.ivf.centroids)

    length(query)==dim||throw(DimensionMismatch("query dim doesnt match centroids"))
    1<=nprobe<=list_count||throw(ArgumentError("nprobe must be between 1 and list count"))
    vector_weight>=0||throw(ArgumentError("vector weight cant be negative"))
    filter_weight>=0||throw(ArgumentError("filter weight cant be negative"))
    vector_weight+filter_weight>0||throw(
        ArgumentError("atleast one weight must be positive"),
    )
    rerank_factor>0||throw(ArgumentError("rerank factor must be positive"))

    distances=centroid_distances(index.ivf, query)

    pool_size=min(list_count, max(nprobe, nprobe*rerank_factor))
    candidate_pool=collect(partialsortperm(distances, 1:pool_size))
    vector_scores=1.0 .- normalized_scores(distances[candidate_pool])
    densities=[
        estimate_list_filter_density(index, list_index, filter) for
        list_index in candidate_pool
    ]
    density_scores=normalized_scores(densities)

    total_weight=vector_weight+filter_weight
    vector_weight/=total_weight
    filter_weight/=total_weight

    scores=Vector{Float64}(undef, pool_size)

    for (position, list_index) in enumerate(candidate_pool)
        scores[position]=vector_weight*vector_scores[position]+filter_weight*density_scores[position]
    end

    order=sortperm(scores; rev = true)
    return candidate_pool[order[1:nprobe]]
end

function select_filter_aware_lists(
    index::FilterAwareIVFIndex,
    query::AbstractVector,
    filter::Union{NamedTuple,FilterExpr};
    k::Int,
    nprobe::Int,
    adaptive::Bool = false,
    max_nprobe::Int = nprobe,
    candidate_multiplier::Float64 = 4.0,
    vector_weight::Float64 = 0.5,
    filter_weight::Float64 = 0.5,
    rerank_factor::Int = 4,
)
    k>0||throw(ArgumentError("k must be positive"))
    nprobe>0||throw(ArgumentError("nprobe must be positive"))
    candidate_multiplier>0||throw(ArgumentError("candidate multiplier must be positive"))
    nprobe<=max_nprobe||throw(ArgumentError("max nprobe cannot be smaller than nprobe"))

    probe_limit=adaptive ? max_nprobe : nprobe
    ordered_lists=rank_filter_aware_lists(
        index,
        query,
        filter;
        nprobe = probe_limit,
        vector_weight = vector_weight,
        filter_weight = filter_weight,
        rerank_factor = rerank_factor,
    )

    adaptive||return ordered_lists

    target_count=max(k, ceil(Int, k*candidate_multiplier))
    estimated_count=0.0
    selected_lists=Int[]

    for list_index in ordered_lists
        push!(selected_lists, list_index)
        estimated_count+=estimate_list_filter_count(index, list_index, filter)

        length(selected_lists)>=nprobe&&estimated_count>=target_count&&break
    end

    return selected_lists
end

function select_filter_aware_candidates(
    index::FilterAwareIVFIndex,
    metadata::AbstractVector,
    query::AbstractVector,
    filter::Union{NamedTuple,FilterExpr};
    k::Int,
    nprobe::Int,
    adaptive::Bool = false,
    max_nprobe::Int = nprobe,
    candidate_multiplier::Float64 = 4.0,
    vector_weight::Float64 = 0.5,
    filter_weight::Float64 = 0.5,
    rerank_factor::Int = 4,
)
    vector_count=sum(length, index.ivf.lists)
    length(metadata)==vector_count||throw(
        DimensionMismatch("metadata count doesnt match index"),
    )

    selected_lists=select_filter_aware_lists(
        index,
        query,
        filter;
        k = k,
        nprobe = nprobe,
        adaptive = adaptive,
        max_nprobe = max_nprobe,
        candidate_multiplier = candidate_multiplier,
        vector_weight = vector_weight,
        filter_weight = filter_weight,
        rerank_factor = rerank_factor,
    )
    candidate_indices=collect_filtered_list_candidates(index, selected_lists, filter)

    return (
        selected_lists = selected_lists,
        visited_indices = candidate_indices,
        candidate_indices = candidate_indices,
    )
end

function search_filter_aware_ivf(
    index::FilterAwareIVFIndex,
    vectors::AbstractMatrix,
    metadata::AbstractVector,
    query::AbstractVector;
    k::Int = 10,
    nprobe::Int = 1,
    metric::Symbol = :cosine,
    filter::Union{NamedTuple,FilterExpr},
    adaptive::Bool = false,
    max_nprobe::Int = nprobe,
    candidate_multiplier::Float64 = 4.0,
    vector_weight::Float64 = 0.5,
    filter_weight::Float64 = 0.5,
    rerank_factor::Int = 4,
    excluded::Union{Nothing,BitVector} = nothing,
)
    validate_ivf_search(index.ivf, vectors, metadata, query)

    selected_lists=select_filter_aware_lists(
        index,
        query,
        filter;
        k = k,
        nprobe = nprobe,
        adaptive = adaptive,
        max_nprobe = max_nprobe,
        candidate_multiplier = candidate_multiplier,
        vector_weight = vector_weight,
        filter_weight = filter_weight,
        rerank_factor = rerank_factor,
    )
    workspace=search_workspace()
    candidate_indices=collect_filtered_list_candidates!(
        workspace.candidate_indices,
        index,
        selected_lists,
        normalize_filter(filter),
    )

    return score_ivf_candidates(
        vectors,
        metadata,
        query,
        candidate_indices;
        k = k,
        metric = metric,
        vector_norms = index.ivf.vector_norms,
        excluded = excluded,
        workspace = workspace,
    )
end
