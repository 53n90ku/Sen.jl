function cosine_similarity(a::AbstractVector, b::AbstractVector)
    length(a)==length(b) || throw(DimensionMismatch("vectors should be of same size"))

    norm_a = sqrt(sum(abs2, a))
    norm_b = sqrt(sum(abs2, b))

    iszero(norm_a) && throw(ArgumentError("first vector should not be zero"))
    iszero(norm_b) && throw(ArgumentError("second vector should not be zero"))

    return Float32(dot_similarity(a, b)/(norm_a*norm_b))
end
