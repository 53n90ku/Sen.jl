struct IVFIndex
    centroids::Matrix{Float32}
    lists::Vector{Vector{Int}}
end

function  squared_distance(a::AbstractVector,b::AbstractVector,)::Float32
    length(a)==length(b)||throw(DimensionMismatch("vectors should have same length"))
    distance = 0.0f0
    for i in eachindex(a,b)
        difference = Float32(a[i])-Float32(b[i])
        distance+=difference*difference
    end
    return distance
end

function train_centroids(vectors::AbstractMatrix;nlists::Int,iterations::Int=20,seed::Int=42,)
    dim,count = size(vectors)

    nlists>0||throw(ArgumentError("nlists must be positive"))
    nlists<=count||throw(ArgumentError("nlists cannot exceed vector count"))
    iterations>0|| throw(ArgumentError("iterations must be positive"))

    rng = MersenneTwister(seed)

    starting_indices=randperm(rng,count)[1:nlists]
    centroids=Matrix{Float32}(vectors[:,starting_indices])
    assignments=Vector{Int}(undef,count)

    for _ in 1:iterations
        for vector_index in 1:count
            vector = @view vectors[:,vector_index]
            best_list =1
            best_distance = typemax(Float32)

            for list_index in 1:nlists
                centroid = @view centroids[:,list_index]
                distance = squared_distance(vector,centroid)

                if distance < best_distance
                    best_distance=distance
                    best_list=list_index
                end
            end

            assignments[vector_index]=best_list
        end

        new_centroids = zeros(Float32,dim,nlists)
        list_counts = zeros(Int,nlists)

        for vector_index in 1:count
            list_index = assignments[vector_index]
            list_counts[list_index]+=1

            for dimension in 1:dim
                new_centroids[dimension,list_index]+=Float32(vectors[dimension,vector_index])
            end
        end

        for list_index in 1:nlists
            if list_counts[list_index]==0
                random_index = rand(rng,1:count)
                new_centroids[:,list_index].=vectors[:,random_index]
            else
                new_centroids[:,list_index]./= list_counts[list_index]
            end
        end
        centroids = new_centroids
    end
    return centroids
end

function nearest_centroid(centroids::AbstractMatrix, vector::AbstractVector,)::Int
    dim, centroid_count = size(centroids)
    length(vector)==dim|| throw(DimensionMismatch("vector dim doesnot match centroids"))
    centroid_count>0|| throw(ArgumentError("centroid cannot be empty"))
    best_list =1
    best_distance=typemax(Float32)

    for list_index in 1:centroid_count
        centroid = @view centroids[:, list_index]
        distance = squared_distance(vector,centroid)

        if distance<best_distance
            best_distance=distance
            best_list=list_index
        end
    end
    return best_list
end

function build_ivf(vectors::AbstractMatrix;nlists::Int,iterations::Int=20,seed::Int = 42,)::IVFIndex
    _,count=size(vectors)

    centroids = train_centroids(vectors,nlists=nlists,iterations=iterations,seed = seed)
    lists = [Int[] for _ in 1:nlists]

    for vector_index in 1:count
        vector = @view vectors[:, vector_index]
        list_index = nearest_centroid(centroids,vector)

        push!(lists[list_index],vector_index)
    end
    return IVFIndex(centroids,lists)
end

function search_ivf(index::IVFIndex,vectors::AbstractMatrix,metadata::AbstractVector,query::AbstractVector;k::Int=10,nprobe::Int=1,metric::Symbol = :cosine,)
    dim, vector_count=size(vectors)
    centroid_dim, list_count = size(index.centroids)

    length(query)==dim || throw(DimensionMismatch("query dim doesnt match vectors"))
    centroid_dim==dim|| throw(DimensionMismatch("centrod dim doent match vectors"))
    length(metadata)==vector_count||throw(DimensionMismatch("metadata cont doent match vectors"))

    k>0 || throw(ArgumentError("k must be postiv"))
    1<=nprobe <=list_count||throw(ArgumentError("nprobe must be between 1 and list count"))

    centroid_distances = Vector{Float32}(undef, list_count)

    for list_index in 1:list_count
        centroid = @view index.centroids[:,list_index]
        centroid_distances[list_index]= squared_distance(query,centroid)
    end

    selected_lists = partialsortperm(centroid_distances,1:nprobe,)

    candidate_indices = Int[]
    for list_index in selected_lists
        append!(candidate_indices,index.lists[list_index])
    end

    isempty(candidate_indices)&& return []

    scores=Vector{Float32}(undef,length(candidate_indices))

    for (position,vector_index) in enumerate(candidate_indices)
        vector = @view vectors[:,vector_index]

        if metric == :cosine
            scores[position]=cosine_similarity(query,vector)
        elseif metric ==:dot
            scores[position]=dot_similarity(query,vector)
        else
            throw(ArgumentError("metric must be cosine or dot"))
        end
    end

    ranked = top_k(scores,min(k,length(scores)))

    return[(
        index = candidate_indices[result.index],
        score=result.score,
        metadata = metadata[candidate_indices[result.index]],
    )
    for result in ranked]
    



end

