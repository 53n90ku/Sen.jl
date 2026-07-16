function merge_topk(
    existing_ids::Vector{Int},
    existing_scores::Vector{Float32},
    new_ids::AbstractVector{<:Integer},
    new_scores::AbstractVector{<:Real},
    k::Int,
)
    ids=vcat(existing_ids, Int.(new_ids))
    scores=vcat(existing_scores, Float32.(new_scores))
    count=min(k, length(scores))
    isempty(scores)&&return (Int[], Float32[])
    positions=partialsortperm(scores, 1:count; rev = true)
    return (ids[positions], scores[positions])
end

function compute_blockwise_groundtruth(
    vectors::AbstractMatrix,
    metadata::AbstractVector,
    queries::AbstractMatrix;
    k::Int = 10,
    metric::Symbol = :cosine,
    filters = nothing,
    block_size::Int = 2_048,
)
    dimension, vector_count=size(vectors)
    query_dimension, query_count=size(queries)
    dimension==query_dimension||throw(DimensionMismatch("queries do not match vectors"))
    length(metadata)==vector_count||throw(
        DimensionMismatch("metadata do not match vectors"),
    )
    k>0||throw(ArgumentError("k must be positive"))
    block_size>0||throw(ArgumentError("block size must be positive"))
    metric in (:cosine, :dot)||throw(ArgumentError("metric must be cosine or dot"))
    query_filters=filters===nothing ? fill(nothing, query_count) : filters
    length(query_filters)==query_count||throw(
        DimensionMismatch("filter count does not match queries"),
    )
    groups=Dict{Any,Vector{Int}}()
    for (query_index, filter) in enumerate(query_filters)
        push!(get!(groups, filter, Int[]), query_index)
    end
    filter_index=build_bitset_index(metadata)
    truth=Vector{Vector{Int}}(undef, query_count)

    for (filter, query_indices) in groups
        mask=filter===nothing ? trues(vector_count) : evaluate_filter(filter_index, filter)
        matching=findall(mask)
        if isempty(matching)
            for query_index in query_indices
                truth[query_index]=Int[]
            end
            continue
        end
        query_matrix=Matrix{Float32}(queries[:, query_indices])
        query_norms=metric===:cosine ? compute_vector_norms(query_matrix) :
                    ones(Float32, length(query_indices))
        top_ids=[Int[] for _ in query_indices]
        top_scores=[Float32[] for _ in query_indices]

        for start = 1:block_size:length(matching)
            stop=min(length(matching), start+block_size-1)
            block_ids=@view matching[start:stop]
            block_vectors=Matrix{Float32}(vectors[:, block_ids])
            scores=transpose(query_matrix)*block_vectors
            vector_norms=metric===:cosine ? compute_vector_norms(block_vectors) :
                         ones(Float32, length(block_ids))

            for local_query in eachindex(query_indices)
                if metric===:cosine
                    denominator=query_norms[local_query]
                    iszero(denominator)&&throw(ArgumentError("query vector cannot be zero"))
                    @inbounds @simd for position in eachindex(block_ids)
                        scores[local_query, position]/=denominator*vector_norms[position]
                    end
                end
                block_count=min(k, length(block_ids))
                positions=partialsortperm(
                    @view(scores[local_query, :]),
                    1:block_count;
                    rev = true,
                )
                ids, saved_scores=merge_topk(
                    top_ids[local_query],
                    top_scores[local_query],
                    block_ids[positions],
                    scores[local_query, positions],
                    k,
                )
                top_ids[local_query]=ids
                top_scores[local_query]=saved_scores
            end
        end

        for (local_query, query_index) in enumerate(query_indices)
            truth[query_index]=top_ids[local_query]
        end
    end
    return truth
end

function load_or_compute_blockwise_groundtruth(
    path::AbstractString,
    manifest::DatasetManifest,
    vectors::AbstractMatrix,
    metadata::AbstractVector,
    queries::AbstractMatrix,
    filters::AbstractVector,
    query_indices::AbstractVector{<:Integer};
    k::Int = 10,
    metric::Symbol = :cosine,
    block_size::Int = 2_048,
)
    signature=groundtruth_cache_signature(
        manifest.preprocessing_hash,
        query_indices,
        k,
        metric,
        "em-blockwise-v1",
    )
    cache_path=joinpath(path, "groundtruth", "em-$(signature)")
    if isfile("$(cache_path).ivecs")&&isfile("$(cache_path).toml")
        return load_groundtruth_cache(
            cache_path;
            dataset_hash = manifest.preprocessing_hash,
            query_indices = query_indices,
            k = k,
            metric = metric,
            filter_name = "em-blockwise-v1",
        )
    end
    println(
        "computing exact filtered groundtruth queries=$(length(query_indices)) vectors=$(size(vectors,2))",
    )
    truth=compute_blockwise_groundtruth(
        vectors,
        metadata,
        queries;
        k = k,
        metric = metric,
        filters = filters,
        block_size = block_size,
    )
    save_groundtruth_cache(
        cache_path,
        truth;
        dataset_hash = manifest.preprocessing_hash,
        query_indices = query_indices,
        k = k,
        metric = metric,
        filter_name = "em-blockwise-v1",
    )
    return truth
end
