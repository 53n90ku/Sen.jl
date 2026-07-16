function validate_excluded(excluded::Union{Nothing,BitVector}, vector_count::Int)
    excluded===nothing||length(excluded)==vector_count||throw(
        DimensionMismatch("excluded count doesnt match vectors"),
    )
    return nothing
end

function exclude_candidates(
    candidate_indices::AbstractVector{<:Integer},
    excluded::Union{Nothing,BitVector},
    vector_count::Int,
)
    validate_excluded(excluded, vector_count)
    excluded===nothing&&return candidate_indices
    return Int[index for index in candidate_indices if !excluded[index]]
end

function search_exact(
    vectors::AbstractMatrix,
    metadata::AbstractVector,
    query::AbstractVector;
    k::Int = 10,
    metric::Symbol = :cosine,
    filter::Union{Nothing,NamedTuple,FilterExpr} = nothing,
    filter_index::Union{Nothing,BitsetIndex} = nothing,
    vector_norms::Union{Nothing,AbstractVector} = nothing,
    excluded::Union{Nothing,BitVector} = nothing,
)
    dim, count=size(vectors)
    length(query)==dim||throw(DimensionMismatch("query dimension doesnt match vectors"))
    length(metadata)==count||throw(DimensionMismatch("metadata count doesnt match vectors"))
    k>0||throw(ArgumentError("k must be positive"))
    normalized_filter=normalize_filter(filter)
    normalized_filter===nothing||filter_index===nothing||filter_index.count==count||throw(
        DimensionMismatch("bitset index count doesnt match vectors"),
    )

    candidate_indices=if normalized_filter===nothing
        1:count
    elseif filter_index!==nothing&&supports_indexed_filter(filter_index, normalized_filter)
        mask=evaluate_filter(filter_index, normalized_filter)
        (index for index in eachindex(mask) if mask[index])
    else
        (index for index = 1:count if matches_filter(metadata[index], normalized_filter))
    end

    return score_ivf_candidates(
        vectors,
        metadata,
        query,
        candidate_indices;
        k = k,
        metric = metric,
        vector_norms = vector_norms,
        excluded = excluded,
    )
end
