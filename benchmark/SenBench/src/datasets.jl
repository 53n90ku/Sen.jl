using Random

function normalize_columns!(vectors::AbstractMatrix)
    for index in axes(vectors,2)
        vector=@view vectors[:,index]
        norm_squared=zero(typeof(abs2(zero(eltype(vector)))))

        for dimension in eachindex(vector)
            norm_squared+=abs2(vector[dimension])
        end

        norm=sqrt(norm_squared)
        norm>0||throw(ArgumentError("vector cannot be zero"))

        for dimension in eachindex(vector)
            vector[dimension]/=norm
        end
    end

    return vectors
end

function generate_synthetic_queries(count::Int,dim::Int;seed::Int=43,)
    count>0||throw(ArgumentError("count must be positive"))
    dim>0||throw(ArgumentError("dim must be positive"))

    rng=MersenneTwister(seed)
    queries=randn(rng,Float32,dim,count)

    return normalize_columns!(queries)
end

function generate_synthetic_dataset(n::Int,dim::Int;seed::Int=42,)
    n>0||throw(ArgumentError("n must be positive"))
    dim>0||throw(ArgumentError("dim must be positive"))

    rng=MersenneTwister(seed)
    vectors=randn(rng,Float32,dim,n)
    normalize_columns!(vectors)

    topics=["systems","machine-learning","databases"]
    languages=["julia","python","rust"]

    metadata=[(
        topic=rand(rng,topics),
        language=rand(rng,languages),
        year=rand(rng,2020:2026),
    ) for _ in 1:n]

    return(vectors=vectors,metadata=metadata,)
end

function sample_cluster(rng::AbstractRNG,weights::AbstractVector{<:Real})
    total=sum(weights)
    total>0||throw(ArgumentError("cluster weights must contain a positive value"))
    target=rand(rng)*total
    cumulative=0.0

    for index in eachindex(weights)
        cumulative+=weights[index]
        target<=cumulative&&return index
    end

    return lastindex(weights)
end

function generate_clustered_dataset(n::Int,dim::Int;clusters::Int=16,cluster_noise::Float64=0.10,cluster_skew::Float64=0.0,seed::Int=42,)
    n>0||throw(ArgumentError("n must be positive"))
    dim>0||throw(ArgumentError("dim must be positive"))
    1<=clusters<=n||throw(ArgumentError("clusters must be between 1 and n"))
    cluster_noise>=0||throw(ArgumentError("cluster noise cannot be negative"))
    cluster_skew>=0||throw(ArgumentError("cluster skew cannot be negative"))

    rng=MersenneTwister(seed)
    centers=randn(rng,Float32,dim,clusters)
    normalize_columns!(centers)
    weights=[1/index^cluster_skew for index in 1:clusters]
    assignments=Vector{Int}(undef,n)
    vectors=Matrix{Float32}(undef,dim,n)

    for vector_index in 1:n
        cluster=vector_index<=clusters ? vector_index : sample_cluster(rng,weights)
        assignments[vector_index]=cluster

        for dimension in 1:dim
            vectors[dimension,vector_index]=centers[dimension,cluster]+Float32(cluster_noise*randn(rng))
        end
    end

    normalize_columns!(vectors)
    metadata=[(cluster=assignments[index],) for index in 1:n]

    return(vectors=vectors,metadata=metadata,centers=centers,assignments=assignments,)
end

function generate_clustered_queries(count::Int,centers::AbstractMatrix;query_noise::Float64=0.10,cluster_skew::Float64=0.0,seed::Int=43,)
    dim,cluster_count=size(centers)
    count>0||throw(ArgumentError("count must be positive"))
    cluster_count>0||throw(ArgumentError("centers cannot be empty"))
    query_noise>=0||throw(ArgumentError("query noise cannot be negative"))
    cluster_skew>=0||throw(ArgumentError("cluster skew cannot be negative"))

    rng=MersenneTwister(seed)
    weights=[1/index^cluster_skew for index in 1:cluster_count]
    assignments=Vector{Int}(undef,count)
    queries=Matrix{Float32}(undef,dim,count)

    for query_index in 1:count
        cluster=sample_cluster(rng,weights)
        assignments[query_index]=cluster

        for dimension in 1:dim
            queries[dimension,query_index]=centers[dimension,cluster]+Float32(query_noise*randn(rng))
        end
    end

    normalize_columns!(queries)
    return(queries=queries,assignments=assignments,)
end

function load_fvecs(path::AbstractString;limit::Union{Nothing,Int}=nothing,normalize::Bool=true,)
    isfile(path)||throw(ArgumentError("fvecs file does not exist"))
    limit===nothing||limit>0||throw(ArgumentError("limit must be positive"))
    filesize(path)>sizeof(Int32)||throw(ArgumentError("fvecs file is empty"))

    return open(path,"r") do io
        dim=Int(read(io,Int32))
        dim>0||throw(ArgumentError("stored vector dimension must be positive"))
        record_size=sizeof(Int32)+dim*sizeof(Float32)
        filesize(path)%record_size==0||throw(ArgumentError("fvecs file size does not match its dimension"))
        available_count=filesize(path)÷record_size
        count=limit===nothing ? available_count : min(limit,available_count)
        vectors=Matrix{Float32}(undef,dim,count)
        seekstart(io)

        for vector_index in 1:count
            stored_dim=Int(read(io,Int32))
            stored_dim==dim||throw(DimensionMismatch("fvecs vectors must have one dimension"))
            read!(io,@view vectors[:,vector_index])
        end

        normalize&&normalize_columns!(vectors)
        return vectors
    end
end
