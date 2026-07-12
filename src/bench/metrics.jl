function recall_at_k(predicted_ids::AbstractVector{<:Integer},truth_ids::AbstractVector{<:Integer},k::Int,)::Float64
    k>0||throw(ArgumentError("k must be positive"))
    predicted_top = predicted_ids[1:min(k,length(predicted_ids))]
    truth_top = truth_ids[1:min(k,length(truth_ids))]
    if isempty(truth_top)
        return isempty(predicted_top) ? 1.0 : 0.0
    end
    matches = length(intersect(Set(predicted_top),Set(truth_top)))
    return matches/length(truth_top)
end
