function compute_groundtruth(vectors::AbstractMatrix, metadata::AbstractVector, queries::AbstractMatrix;k::Int=10,metric::Symbol=:cosine,filters=nothing,)
    vector_dim,_=size(vectors)
    query_dim,query_count=size(queries)

    vector_dim==query_dim||throw(DimensionMismatch("doesnt match"))
    query_filters = if filters===nothing 
        fill(nothing,query_count)
    else 
        length(filters)==query_count ||throw(DimensionMismatch("filter count dont match query count"))
        filters
    end

    filter_index=build_bitset_index(metadata)
    groundtruth=Vector{Vector{Int}}(undef,query_count)

    for query_index in 1:query_count
        query = @view queries[:,query_index]

        results = search_exact(vectors,metadata, query;k=k,metric=metric,filter=query_filters[query_index],filter_index=filter_index,)
        groundtruth[query_index]=[
            result.index for result in results
        ]
    end
    return groundtruth
end

    