function validate_excluded(excluded::Union{Nothing,BitVector},vector_count::Int)
    excluded===nothing||length(excluded)==vector_count||throw(DimensionMismatch("excluded count doesnt match vectors"))
    return nothing
end

function exclude_candidates(candidate_indices::AbstractVector{<:Integer},excluded::Union{Nothing,BitVector},vector_count::Int)
    validate_excluded(excluded,vector_count)
    excluded===nothing&&return candidate_indices
    return Int[index for index in candidate_indices if !excluded[index]]
end

function search_exact(vectors::AbstractMatrix,metadata::AbstractVector,query::AbstractVector;k::Int=10,metric::Symbol=:cosine,filter::Union{Nothing,NamedTuple}=nothing,filter_index::Union{Nothing,BitsetIndex}=nothing,vector_norms::Union{Nothing,AbstractVector}=nothing,excluded::Union{Nothing,BitVector}=nothing,)
    dim,count=size(vectors)
    length(query)==dim||throw(DimensionMismatch("query dimension doesnt match vectors"))
    length(metadata)==count||throw(DimensionMismatch("metadata count doesnt match vectors"))
    k>0||throw(ArgumentError("k must be positive"))

    candidate_indices=if filter===nothing
        collect(1:count)
    elseif filter_index!==nothing
        filter_index.count==count||throw(DimensionMismatch("bitset index count doesnt match vectors"))
        findall(evaluate_filter(filter_index,filter))
    else
        Int[index for index in 1:count if matches_filter(metadata[index],filter)]
    end

    return score_ivf_candidates(vectors,metadata,query,candidate_indices;k=k,metric=metric,vector_norms=vector_norms,excluded=excluded,)
end
