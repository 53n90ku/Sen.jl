using Random

function generate_test_clusters(count::Int,dim::Int,clusters::Int;noise::Float64,seed::Int,)
    count>0||throw(ArgumentError("count must be positive"))
    1<=clusters<=dim||throw(ArgumentError("clusters must be between 1 and dimension"))
    centers=zeros(Float32,dim,clusters)

    for cluster in 1:clusters
        centers[cluster,cluster]=1.0f0
    end

    vectors=Matrix{Float32}(undef,dim,count)
    rng=MersenneTwister(seed)

    for index in 1:count
        cluster=mod1(index,clusters)
        vectors[:,index].=centers[:,cluster].+Float32(noise).*randn(rng,Float32,dim)
        vectors[:,index]./=sqrt(sum(abs2,@view vectors[:,index]))
    end

    return(vectors=vectors,centers=centers,)
end

function test_recall_at_k(predicted::AbstractVector{<:Integer},truth::AbstractVector{<:Integer},k::Int)
    truth_top=truth[1:min(k,length(truth))]
    predicted_top=predicted[1:min(k,length(predicted))]
    isempty(truth_top)&&return isempty(predicted_top) ? 1.0 : 0.0
    return length(intersect(Set(predicted_top),Set(truth_top)))/length(truth_top)
end
