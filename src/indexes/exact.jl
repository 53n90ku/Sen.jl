function search_exact(vectors::AbstractMatrix, metadata::AbstractVector,query::AbstractVector; k::Int=10,metric::Symbol =:cosine,filter::Union{Nothing,NamedTuple}=nothing,filter_index::Union{Nothing,BitsetIndex}=nothing,)
    dim,count= size(vectors)
    length(query)==dim || throw(DimensionMismatch("query dimension dont match vectors"))
    length(metadata)==count || throw(DimensionMismatch("metadata count dont match vectors count"))
    k>0|| throw(ArgumentError("k must be positive"))

    candidate_indices=if filter===nothing
        collect(1:count)
    elseif filter_index!==nothing
        filter_index.count==count||throw(DimensionMismatch("bitset index count dont match the vectors"))
        findall(evaluate_filter(filter_index,filter))
    else [
        i for i in 1:count
        if matches_filter(metadata[i],filter)
    ]
        end
        
    isempty(candidate_indices)&&return []
    scores = Vector{Float32}(undef,length(candidate_indices))

    for (position,index) in enumerate(candidate_indices)
        vector = @view vectors[:,index]

        if metric == :cosine
            scores[position]=cosine_similarity(query,vector)
        elseif metric ==:dot
            scores[position]=dot_similarity(query,vector)
        else throw(ArgumentError("metric mus be either cosine or dot"))
        end
    end

    ranked = top_k(scores,min(k,length(scores)))

    return [(
        index = candidate_indices[result.index],
        score=result.score,
        metadata=metadata[candidate_indices[result.index]],
            
    )
    for result in ranked]
end
    