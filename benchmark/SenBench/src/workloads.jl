using Random

function generate_filter_order(vectors::AbstractMatrix;workload::Symbol=:random,seed::Int=42,skew::Float64=4.0,)
    dim,count=size(vectors)
    count>0||throw(ArgumentError("vectors cannot be empty"))
    workload in (:random,:correlated,:anticorrelated,:skewed)||throw(ArgumentError("workload must be random, correlated, anticorrelated or skewed"))
    skew>0||throw(ArgumentError("skew must be positive"))

    rng=MersenneTwister(seed)
    workload===:random&&return randperm(rng,count)

    direction=randn(rng,Float32,dim)
    direction./=sqrt(sum(abs2,direction))
    projections=Vector{Float64}(undef,count)

    for vector_index in 1:count
        vector=@view vectors[:,vector_index]
        projections[vector_index]=dot_similarity(direction,vector)
    end

    if workload===:correlated
        return collect(sortperm(projections;rev=true,))
    elseif workload===:anticorrelated
        return collect(sortperm(projections;rev=false,))
    end

    scores=[skew*projections[index]+randn(rng) for index in 1:count]
    return collect(sortperm(scores;rev=true,))
end

function generate_filter_metadata(vectors::AbstractMatrix,selectivity::Float64;workload::Symbol=:random,seed::Int=42,skew::Float64=4.0,)
    order=generate_filter_order(vectors;workload=workload,seed=seed,skew=skew,)
    return generate_selectivity_metadata(order,selectivity)
end
