function top_k(scores::AbstractVector,k::Int)
    k>0|| throw(ArgumentError("k should be positive"))
    k<=length(scores)||throw(ArgumentError(" k cant exceed number of scores"))

    indices = partialsortperm(scores,1:k;rev =true)
    return[(
    index=index,score = Float32(scores[index]),
    )
    for index in indices]
end

