function dot_similarity(a::AbstractVector,b::AbstractVector)
    length(a)==length(b) || throw(DimensionMismatch("vectors must have same length"))

    score = 0.0f0
    for i in eachindex(a,b)
        score+=Float32(a[i])*Float32(b[i])
    end

    return score
end