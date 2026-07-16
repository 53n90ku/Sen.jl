using Random
using LinearAlgebra

struct IVFIndex
    centroids::Matrix{Float32}
    lists::Vector{Vector{Int}}
    vector_norms::Vector{Float32}
    metric::Symbol
    routing::Symbol
    list_radii::Vector{Float32}
    list_cos_radii::Vector{Float32}
    list_sin_radii::Vector{Float32}
end

IVFIndex(centroids::Matrix{Float32},lists::Vector{Vector{Int}})=IVFIndex(centroids,lists,Float32[],:cosine,:cosine,fill(Float32(pi),length(lists)),fill(-1.0f0,length(lists)),zeros(Float32,length(lists)))
IVFIndex(centroids::Matrix{Float32},lists::Vector{Vector{Int}},vector_norms::Vector{Float32})=IVFIndex(centroids,lists,vector_norms,:cosine,:cosine,fill(Float32(pi),length(lists)),fill(-1.0f0,length(lists)),zeros(Float32,length(lists)))
IVFIndex(centroids::Matrix{Float32},lists::Vector{Vector{Int}},vector_norms::Vector{Float32},metric::Symbol,list_radii::Vector{Float32},list_cos_radii::Vector{Float32},list_sin_radii::Vector{Float32})=IVFIndex(centroids,lists,vector_norms,metric,metric,list_radii,list_cos_radii,list_sin_radii)

function vector_norm(vector::AbstractVector)::Float32
    squared_norm=0.0f0

    @inbounds @simd for index in eachindex(vector)
        value=Float32(vector[index])
        squared_norm+=value*value
    end

    return sqrt(squared_norm)
end

function compute_vector_norms(vectors::AbstractMatrix)
    _,count=size(vectors)
    norms=Vector{Float32}(undef,count)

    for vector_index in 1:count
        squared_norm=0.0f0

        @inbounds @simd for dimension in axes(vectors,1)
            value=Float32(vectors[dimension,vector_index])
            squared_norm+=value*value
        end

        norms[vector_index]=sqrt(squared_norm)
    end

    return norms
end

function column_dot(query::AbstractVector,vectors::AbstractMatrix,vector_index::Int)
    score=0.0f0

    @inbounds @simd for dimension in eachindex(query)
        score+=Float32(query[dimension])*Float32(vectors[dimension,vector_index])
    end

    return score
end

function column_norm(vectors::AbstractMatrix,vector_index::Int)
    squared_norm=0.0f0

    @inbounds @simd for dimension in axes(vectors,1)
        value=Float32(vectors[dimension,vector_index])
        squared_norm+=value*value
    end

    return sqrt(squared_norm)
end

function squared_distance(a::AbstractVector,b::AbstractVector,)::Float32
    length(a)==length(b)||throw(DimensionMismatch("vectors should have same length"))

    distance=0.0f0
    for index in eachindex(a,b)
        difference=Float32(a[index])-Float32(b[index])
        distance+=difference*difference
    end

    return distance
end

function normalize_vector!(vector::AbstractVector)
    norm=vector_norm(vector)
    iszero(norm)&&throw(ArgumentError("vector cannot be zero"))
    vector./=norm
    return vector
end

function centroid_score(centroids::AbstractMatrix,list_index::Int,vector::AbstractVector,metric::Symbol)
    if metric===:cosine||metric===:dot
        score=0.0f0

        @inbounds @simd for dimension in eachindex(vector)
            score+=Float32(vector[dimension])*centroids[dimension,list_index]
        end

        if metric===:cosine
            vector_length=vector_norm(vector)
            iszero(vector_length)&&throw(ArgumentError("vector cannot be zero"))
            return score/vector_length
        end

        return score
    end

    throw(ArgumentError("metric must be cosine or dot"))
end

function nearest_centroid_score(centroids::AbstractMatrix,vector::AbstractVector,metric::Symbol)
    _,centroid_count=size(centroids)
    best_list=1
    best_score=-typemax(Float32)

    for list_index in 1:centroid_count
        score=centroid_score(centroids,list_index,vector,metric)

        if score>best_score
            best_score=score
            best_list=list_index
        end
    end

    return(best_list=best_list,score=best_score,)
end

function sample_centroid_position(rng::AbstractRNG,losses::AbstractVector{<:Real},selected::BitVector)
    available=Int[index for index in eachindex(losses) if !selected[index]]
    isempty(available)&&throw(ArgumentError("no centroid positions remain"))
    total=sum(Float64(losses[index]) for index in available)
    total>0||return rand(rng,available)
    target=rand(rng)*total
    cumulative=0.0

    for index in available
        cumulative+=losses[index]
        target<=cumulative&&return index
    end

    return last(available)
end

function initialize_centroids(vectors::AbstractMatrix,training_indices::AbstractVector{<:Integer},nlists::Int,metric::Symbol,rng::AbstractRNG)
    dim=size(vectors,1)
    training_count=length(training_indices)
    centroids=Matrix{Float32}(undef,dim,nlists)

    if metric===:dot
        positions=randperm(rng,training_count)[1:nlists]

        for list_index in 1:nlists
            centroids[:,list_index].=vectors[:,training_indices[positions[list_index]]]
        end

        return centroids
    end

    selected=falses(training_count)
    minimum_losses=fill(Inf32,training_count)
    selected_position=rand(rng,1:training_count)

    for list_index in 1:nlists
        selected[selected_position]=true
        centroids[:,list_index].=vectors[:,training_indices[selected_position]]
        metric===:cosine&&normalize_vector!(@view centroids[:,list_index])

        list_index==nlists&&break

        for position in 1:training_count
            if selected[position]
                minimum_losses[position]=0.0f0
                continue
            end

            vector=@view vectors[:,training_indices[position]]
            score=centroid_score(centroids,list_index,vector,metric)
            loss=max(0.0f0,1.0f0-score)
            minimum_losses[position]=min(minimum_losses[position],loss)
        end

        selected_position=sample_centroid_position(rng,minimum_losses,selected)
    end

    return centroids
end

function cosine_assignments(centroids::AbstractMatrix,vectors::AbstractMatrix)
    dimension,list_count=size(centroids)
    size(vectors,1)==dimension||throw(DimensionMismatch("vectors do not match centroids"))
    vector_count=size(vectors,2)
    scores=Matrix{Float32}(undef,list_count,vector_count)
    mul!(scores,transpose(centroids),vectors)
    assignments=Vector{Int}(undef,vector_count)
    best_scores=Vector{Float32}(undef,vector_count)

    for vector_index in 1:vector_count
        best_list=1
        best_score=scores[1,vector_index]
        @inbounds for list_index in 2:list_count
            score=scores[list_index,vector_index]
            if score>best_score
                best_score=score
                best_list=list_index
            end
        end
        assignments[vector_index]=best_list
        best_scores[vector_index]=best_score
    end

    return(assignments=assignments,best_scores=best_scores,)
end

function initialize_cosine_centroids(training_vectors::Matrix{Float32},nlists::Int,rng::AbstractRNG,vector_norms::AbstractVector)
    dimension,training_count=size(training_vectors)
    centroids=Matrix{Float32}(undef,dimension,nlists)
    selected=falses(training_count)
    minimum_losses=fill(Inf32,training_count)
    scores=Vector{Float32}(undef,training_count)
    selected_position=rand(rng,1:training_count)

    for list_index in 1:nlists
        selected[selected_position]=true
        centroids[:,list_index].=training_vectors[:,selected_position]
        normalize_vector!(@view centroids[:,list_index])
        list_index==nlists&&break
        mul!(scores,transpose(training_vectors),@view centroids[:,list_index])

        @inbounds @simd for position in 1:training_count
            if selected[position]
                minimum_losses[position]=0.0f0
            else
                similarity=scores[position]/vector_norms[position]
                minimum_losses[position]=min(minimum_losses[position],max(0.0f0,1.0f0-similarity))
            end
        end

        selected_position=sample_centroid_position(rng,minimum_losses,selected)
    end

    return centroids
end

function train_cosine_centroids(vectors::AbstractMatrix,training_indices::AbstractVector{<:Integer},nlists::Int,iterations::Int,restarts::Int,rng::AbstractRNG)
    dimension=size(vectors,1)
    training_vectors=Matrix{Float32}(vectors[:,training_indices])
    vector_norms=compute_vector_norms(training_vectors)
    any(iszero,vector_norms)&&throw(ArgumentError("cosine training vectors cannot be zero"))
    training_count=size(training_vectors,2)
    best_centroids=Matrix{Float32}(undef,dimension,nlists)
    best_loss=Inf

    for _ in 1:restarts
        centroids=initialize_cosine_centroids(training_vectors,nlists,rng,vector_norms)

        for _ in 1:iterations
            assignments=cosine_assignments(centroids,training_vectors).assignments
            new_centroids=zeros(Float32,dimension,nlists)
            list_counts=zeros(Int,nlists)

            for vector_index in 1:training_count
                list_index=assignments[vector_index]
                list_counts[list_index]+=1
                @inbounds @simd for axis in 1:dimension
                    new_centroids[axis,list_index]+=training_vectors[axis,vector_index]
                end
            end

            for list_index in 1:nlists
                collapsed=iszero(vector_norm(@view new_centroids[:,list_index]))
                if list_counts[list_index]==0||collapsed
                    new_centroids[:,list_index].=training_vectors[:,rand(rng,1:training_count)]
                end
                normalize_vector!(@view new_centroids[:,list_index])
            end

            centroids=new_centroids
        end

        nearest=cosine_assignments(centroids,training_vectors)
        loss=sum(1.0-nearest.best_scores[index]/vector_norms[index] for index in 1:training_count)
        if loss<best_loss
            best_loss=loss
            best_centroids.=centroids
        end
    end

    return best_centroids
end

function train_centroids(vectors::AbstractMatrix;nlists::Int,iterations::Int=20,seed::Int=42,metric::Symbol=:cosine,restarts::Int=1,training_count::Int=size(vectors,2),)
    dim,count=size(vectors)

    nlists>0||throw(ArgumentError("nlists must be positive"))
    nlists<=count||throw(ArgumentError("nlists cannot exceed vector count"))
    iterations>0||throw(ArgumentError("iterations must be positive"))
    metric in (:cosine,:dot)||throw(ArgumentError("metric must be cosine or dot"))
    restarts>0||throw(ArgumentError("restarts must be positive"))
    nlists<=training_count<=count||throw(ArgumentError("training count must be between nlists and vector count"))

    rng=MersenneTwister(seed)
    training_indices=training_count==count ? collect(1:count) : randperm(rng,count)[1:training_count]

    if metric===:cosine
        return train_cosine_centroids(vectors,training_indices,nlists,iterations,restarts,rng)
    end

    assignments=Vector{Int}(undef,training_count)
    best_centroids=Matrix{Float32}(undef,dim,nlists)
    best_loss=Inf

    for _ in 1:restarts
        centroids=initialize_centroids(vectors,training_indices,nlists,metric,rng)

        for _ in 1:iterations
            for(position,vector_index) in enumerate(training_indices)
                vector=@view vectors[:,vector_index]
                assignments[position]=nearest_centroid_score(centroids,vector,metric).best_list
            end

            new_centroids=zeros(Float32,dim,nlists)
            list_counts=zeros(Int,nlists)

            for(position,vector_index) in enumerate(training_indices)
                list_index=assignments[position]
                list_counts[list_index]+=1

                @inbounds @simd for dimension in 1:dim
                    new_centroids[dimension,list_index]+=Float32(vectors[dimension,vector_index])
                end
            end

            for list_index in 1:nlists
                collapsed=metric===:cosine&&iszero(vector_norm(@view new_centroids[:,list_index]))

                if list_counts[list_index]==0||collapsed
                    random_position=rand(rng,1:training_count)
                    new_centroids[:,list_index].=vectors[:,training_indices[random_position]]
                elseif metric===:dot
                    new_centroids[:,list_index]./=list_counts[list_index]
                end

                metric===:cosine&&normalize_vector!(@view new_centroids[:,list_index])
            end

            centroids=new_centroids
        end

        loss=0.0

        for vector_index in training_indices
            vector=@view vectors[:,vector_index]
            nearest=nearest_centroid_score(centroids,vector,metric)
            loss+=metric===:cosine ? 1.0-nearest.score : -nearest.score
        end

        if loss<best_loss
            best_loss=loss
            best_centroids.=centroids
        end
    end

    return best_centroids
end

function nearest_centroid(centroids::AbstractMatrix,vector::AbstractVector;metric::Symbol=:cosine,)::Int
    dim,centroid_count=size(centroids)
    length(vector)==dim||throw(DimensionMismatch("vector dim doesnt match centroids"))
    centroid_count>0||throw(ArgumentError("centroids cannot be empty"))
    metric in (:cosine,:dot)||throw(ArgumentError("metric must be cosine or dot"))

    return nearest_centroid_score(centroids,vector,metric).best_list
end

function compute_list_radii(centroids::AbstractMatrix,lists::Vector{Vector{Int}},vectors::AbstractMatrix,vector_norms::AbstractVector,metric::Symbol)
    radii=metric===:cosine ? zeros(Float32,length(lists)) : fill(Float32(pi),length(lists))
    metric===:cosine||return radii

    for list_index in eachindex(lists)
        centroid=@view centroids[:,list_index]

        for vector_index in lists[list_index]
            norm=Float32(vector_norms[vector_index])
            iszero(norm)&&throw(ArgumentError("stored vector cannot be zero"))
            similarity=clamp(column_dot(centroid,vectors,vector_index)/norm,-1.0f0,1.0f0)
            angle=acos(similarity)
            radii[list_index]=max(radii[list_index],angle)
        end

        radii[list_index]=min(Float32(pi),radii[list_index]+1.0f-6)
    end

    return radii
end

function build_ivf(vectors::AbstractMatrix;nlists::Int,iterations::Int=20,seed::Int=42,metric::Symbol=:cosine,restarts::Int=1,training_count::Int=size(vectors,2),)::IVFIndex
    _,count=size(vectors)
    centroids=train_centroids(vectors;nlists=nlists,iterations=iterations,seed=seed,metric=metric,restarts=restarts,training_count=training_count,)
    lists=[Int[] for _ in 1:nlists]

    if metric===:cosine
        assignments=cosine_assignments(centroids,vectors).assignments
        for vector_index in 1:count
            push!(lists[assignments[vector_index]],vector_index)
        end
    else
        for vector_index in 1:count
            vector=@view vectors[:,vector_index]
            list_index=nearest_centroid(centroids,vector;metric=metric,)
            push!(lists[list_index],vector_index)
        end
    end

    vector_norms=compute_vector_norms(vectors)
    list_radii=compute_list_radii(centroids,lists,vectors,vector_norms,metric)
    list_cos_radii=cos.(list_radii)
    list_sin_radii=sin.(list_radii)

    return IVFIndex(centroids,lists,vector_norms,metric,metric,list_radii,list_cos_radii,list_sin_radii)
end

function centroid_distances!(distances::Vector{Float32},index::IVFIndex,query::AbstractVector)
    dim,list_count=size(index.centroids)
    length(query)==dim||throw(DimensionMismatch("query dimension doesnt match centroids"))

    resize!(distances,list_count)
    query_norm=index.routing===:cosine ? vector_norm(query) : 1.0f0
    index.routing===:cosine&&iszero(query_norm)&&throw(ArgumentError("query vector cannot be zero"))

    for list_index in 1:list_count
        if index.routing===:cosine
            similarity=0.0f0

            @inbounds @simd for dimension in 1:dim
                similarity+=Float32(query[dimension])*index.centroids[dimension,list_index]
            end

            distances[list_index]=1.0f0-similarity/query_norm
        elseif index.routing===:dot
            score=0.0f0

            @inbounds @simd for dimension in 1:dim
                score+=Float32(query[dimension])*index.centroids[dimension,list_index]
            end

            distances[list_index]=-score
        elseif index.routing===:euclidean
            distance=0.0f0

            @inbounds @simd for dimension in 1:dim
                difference=Float32(query[dimension])-index.centroids[dimension,list_index]
                distance+=difference*difference
            end

            distances[list_index]=distance
        else
            throw(ArgumentError("index routing must be cosine, dot or euclidean"))
        end
    end

    return distances
end

function centroid_distances(index::IVFIndex,query::AbstractVector)
    return centroid_distances!(Vector{Float32}(undef,size(index.centroids,2)),index,query)
end

function list_score_upper_bound(index::IVFIndex,query::AbstractVector,list_index::Int)
    index.metric===:cosine||throw(ArgumentError("score bounds require a cosine index"))
    1<=list_index<=length(index.lists)||throw(BoundsError(index.lists,list_index))
    isempty(index.lists[list_index])&&return -Inf32

    length(query)==size(index.centroids,1)||throw(DimensionMismatch("query dimension doesnt match centroids"))
    query_norm=vector_norm(query)
    iszero(query_norm)&&throw(ArgumentError("query vector cannot be zero"))
    return list_score_upper_bound(index,query,list_index,query_norm)
end

function list_score_upper_bound(index::IVFIndex,query::AbstractVector,list_index::Int,query_norm::Float32)
    isempty(index.lists[list_index])&&return -Inf32
    similarity=0.0f0

    @inbounds @simd for dimension in eachindex(query)
        similarity+=Float32(query[dimension])*index.centroids[dimension,list_index]
    end

    cosine_angle=clamp(similarity/query_norm,-1.0f0,1.0f0)
    cosine_radius=index.list_cos_radii[list_index]
    cosine_angle>=cosine_radius&&return 1.0f0
    sine_angle=sqrt(max(0.0f0,1.0f0-cosine_angle*cosine_angle))
    bound=cosine_angle*cosine_radius+sine_angle*index.list_sin_radii[list_index]
    return min(1.0f0,bound+2.0f-6)
end

function list_score_upper_bounds(index::IVFIndex,query::AbstractVector)
    index.metric===:cosine||throw(ArgumentError("score bounds require a cosine index"))
    length(query)==size(index.centroids,1)||throw(DimensionMismatch("query dimension doesnt match centroids"))
    query_norm=vector_norm(query)
    iszero(query_norm)&&throw(ArgumentError("query vector cannot be zero"))
    return[list_score_upper_bound(index,query,list_index,query_norm) for list_index in eachindex(index.lists)]
end

function rank_ivf_lists!(selected_lists::Vector{Int},index::IVFIndex,query::AbstractVector,nprobe::Int,workspace::SearchWorkspace)
    _,list_count=size(index.centroids)

    1<=nprobe<=list_count||throw(ArgumentError("nprobe must be between 1 and list count"))
    distances=centroid_distances!(workspace.centroid_distances,index,query)
    reset_top_candidates!(workspace,nprobe)

    for list_index in eachindex(distances)
        add_top_candidate!(workspace,list_index,-distances[list_index],list_index,nprobe)
    end

    sort_top_candidates!(workspace)
    empty!(selected_lists)
    sizehint!(selected_lists,nprobe)
    append!(selected_lists,workspace.heap_indices)
    return selected_lists
end

function rank_ivf_lists(index::IVFIndex,query::AbstractVector;nprobe::Int,)
    workspace=SearchWorkspace()
    return copy(rank_ivf_lists!(workspace.selected_lists,index,query,nprobe,workspace))
end

function collect_ivf_candidates!(candidate_indices::Vector{Int},index::IVFIndex,selected_lists::AbstractVector{<:Integer})
    empty!(candidate_indices)
    sizehint!(candidate_indices,sum(length(index.lists[list_index]) for list_index in selected_lists))

    for list_index in selected_lists
        1<=list_index<=length(index.lists)||throw(BoundsError(index.lists,list_index))
        append!(candidate_indices,index.lists[list_index])
    end

    return candidate_indices
end

function collect_ivf_candidates(index::IVFIndex,selected_lists::AbstractVector{<:Integer})
    candidate_indices=Int[]
    return collect_ivf_candidates!(candidate_indices,index,selected_lists)
end

function _filter_ivf_candidates(candidate_indices::AbstractVector{<:Integer},metadata::AbstractVector,filter::FilterExpr;filter_index::Union{Nothing,BitsetIndex}=nothing,)
    filter_index===nothing||filter_index.count==length(metadata)||throw(DimensionMismatch("filter index count doesnt match metadata"))

    if filter_index===nothing||!supports_indexed_filter(filter_index,filter)
        return Int[index for index in candidate_indices if matches_filter(metadata[index],filter)]
    end

    mask=evaluate_filter(filter_index,filter)
    return Int[index for index in candidate_indices if mask[index]]
end

function filter_ivf_candidates(candidate_indices::AbstractVector{<:Integer},metadata::AbstractVector,filter::Union{NamedTuple,FilterExpr};filter_index::Union{Nothing,BitsetIndex}=nothing,)
    return _filter_ivf_candidates(candidate_indices,metadata,normalize_filter(filter);filter_index=filter_index,)
end

function filter_ivf_candidates!(filtered_indices::Vector{Int},candidate_indices::AbstractVector{<:Integer},metadata::AbstractVector,filter::FilterExpr;filter_index::Union{Nothing,BitsetIndex}=nothing,)
    filter_index===nothing||filter_index.count==length(metadata)||throw(DimensionMismatch("filter index count doesnt match metadata"))
    empty!(filtered_indices)

    if filter_index===nothing||!supports_indexed_filter(filter_index,filter)
        for index in candidate_indices
            matches_filter(metadata[index],filter)&&push!(filtered_indices,index)
        end
    else
        mask=evaluate_filter(filter_index,filter)

        for index in candidate_indices
            mask[index]&&push!(filtered_indices,index)
        end
    end

    return filtered_indices
end

function prepare_query_buffer!(workspace::SearchWorkspace,query::AbstractVector)
    resize!(workspace.query_buffer,length(query))

    @inbounds @simd for index in eachindex(query)
        workspace.query_buffer[index]=Float32(query[index])
    end

    return workspace.query_buffer
end

@inline function candidate_score(vectors::AbstractMatrix,query::AbstractVector,vector_index::Int,metric::Symbol,query_norm::Float32,vector_norms::Union{Nothing,AbstractVector})
    score=column_dot(query,vectors,vector_index)

    if metric===:cosine
        stored_norm=vector_norms===nothing||isempty(vector_norms) ? column_norm(vectors,vector_index) : Float32(vector_norms[vector_index])
        iszero(stored_norm)&&throw(ArgumentError("stored vector cannot be zero"))
        score/=query_norm*stored_norm
    end

    return score
end

function score_candidate_stream!(workspace::SearchWorkspace,vectors::AbstractMatrix,query::AbstractVector,candidate_indices;k::Int,metric::Symbol,vector_norms::Union{Nothing,AbstractVector},excluded::Union{Nothing,BitVector})
    query_norm=metric===:cosine ? vector_norm(query) : 1.0f0
    metric===:cosine&&iszero(query_norm)&&throw(ArgumentError("query vector cannot be zero"))
    order=0

    for raw_index in candidate_indices
        vector_index=Int(raw_index)
        excluded!==nothing&&excluded[vector_index]&&continue
        order+=1
        score=candidate_score(vectors,query,vector_index,metric,query_norm,vector_norms)
        add_top_candidate!(workspace,vector_index,score,order,k)
    end

    return workspace
end

function score_contiguous_candidates!(workspace::SearchWorkspace,vectors::AbstractMatrix,query::AbstractVector,candidate_indices::AbstractUnitRange{<:Integer};k::Int,metric::Symbol,vector_norms::Union{Nothing,AbstractVector},excluded::Union{Nothing,BitVector})
    query_buffer=prepare_query_buffer!(workspace,query)
    query_norm=metric===:cosine ? vector_norm(query_buffer) : 1.0f0
    metric===:cosine&&iszero(query_norm)&&throw(ArgumentError("query vector cannot be zero"))
    resize!(workspace.block_scores,SEARCH_SCORE_BLOCK_SIZE)
    order=0
    first_index=Int(first(candidate_indices))
    last_index=Int(last(candidate_indices))

    for block_start in first_index:SEARCH_SCORE_BLOCK_SIZE:last_index
        block_stop=min(last_index,block_start+SEARCH_SCORE_BLOCK_SIZE-1)
        block_length=block_stop-block_start+1
        block_vectors=@view vectors[:,block_start:block_stop]
        block_scores=@view workspace.block_scores[1:block_length]
        mul!(block_scores,transpose(block_vectors),query_buffer)

        for offset in 1:block_length
            vector_index=block_start+offset-1
            excluded!==nothing&&excluded[vector_index]&&continue
            order+=1
            score=block_scores[offset]

            if metric===:cosine
                stored_norm=vector_norms===nothing||isempty(vector_norms) ? column_norm(vectors,vector_index) : Float32(vector_norms[vector_index])
                iszero(stored_norm)&&throw(ArgumentError("stored vector cannot be zero"))
                score/=query_norm*stored_norm
            end

            add_top_candidate!(workspace,vector_index,score,order,k)
        end
    end

    return workspace
end

function top_candidate_results(metadata::AbstractVector,workspace::SearchWorkspace)
    sort_top_candidates!(workspace)
    return[(index=workspace.heap_indices[position],score=workspace.heap_scores[position],metadata=metadata[workspace.heap_indices[position]],) for position in eachindex(workspace.heap_scores)]
end

function ensure_batch_search_workspace!(workspace::BatchSearchWorkspace,dim::Int,query_count::Int,k::Int)
    if size(workspace.score_block,1)<SEARCH_SCORE_BLOCK_SIZE||size(workspace.score_block,2)<SEARCH_QUERY_BLOCK_SIZE
        workspace.score_block=Matrix{Float32}(undef,SEARCH_SCORE_BLOCK_SIZE,SEARCH_QUERY_BLOCK_SIZE)
    end

    if size(workspace.query_buffer,1)<dim||size(workspace.query_buffer,2)<SEARCH_QUERY_BLOCK_SIZE
        workspace.query_buffer=Matrix{Float32}(undef,dim,SEARCH_QUERY_BLOCK_SIZE)
    end

    resize!(workspace.query_norms,query_count)

    while length(workspace.query_workspaces)<query_count
        push!(workspace.query_workspaces,SearchWorkspace())
    end

    for query_index in 1:query_count
        reset_top_candidates!(workspace.query_workspaces[query_index],k)
    end

    return workspace
end

function search_exact_batch(vectors::AbstractMatrix,metadata::AbstractVector,queries::AbstractMatrix;k::Int=10,metric::Symbol=:cosine,vector_norms::Union{Nothing,AbstractVector}=nothing,excluded::Union{Nothing,BitVector}=nothing,workspace::BatchSearchWorkspace=batch_search_workspace(),)
    dim,vector_count=size(vectors)
    query_dim,query_count=size(queries)
    query_dim==dim||throw(DimensionMismatch("query dimensions dont match vectors"))
    length(metadata)==vector_count||throw(DimensionMismatch("metadata count doesnt match vectors"))
    k>0||throw(ArgumentError("k must be positive"))
    metric in (:cosine,:dot)||throw(ArgumentError("metric must be cosine or dot"))
    validate_excluded(excluded,vector_count)
    vector_norms!==nothing&&!isempty(vector_norms)&&length(vector_norms)!=vector_count&&throw(DimensionMismatch("vector norm count doesnt match vectors"))
    ensure_batch_search_workspace!(workspace,dim,query_count,k)

    for query_start in 1:SEARCH_QUERY_BLOCK_SIZE:query_count
        query_stop=min(query_count,query_start+SEARCH_QUERY_BLOCK_SIZE-1)
        query_length=query_stop-query_start+1

        for local_query in 1:query_length
            query_index=query_start+local_query-1
            squared_norm=0.0f0

            @inbounds @simd for axis in 1:dim
                value=Float32(queries[axis,query_index])
                workspace.query_buffer[axis,local_query]=value
                squared_norm+=value*value
            end

            workspace.query_norms[query_index]=metric===:cosine ? sqrt(squared_norm) : 1.0f0
            metric===:cosine&&iszero(workspace.query_norms[query_index])&&throw(ArgumentError("query vector cannot be zero"))
        end

        query_buffer=@view workspace.query_buffer[1:dim,1:query_length]

        for block_start in 1:SEARCH_SCORE_BLOCK_SIZE:vector_count
            block_stop=min(vector_count,block_start+SEARCH_SCORE_BLOCK_SIZE-1)
            block_length=block_stop-block_start+1
            block_vectors=@view vectors[:,block_start:block_stop]
            block_scores=@view workspace.score_block[1:block_length,1:query_length]
            mul!(block_scores,transpose(block_vectors),query_buffer)

            for local_vector in 1:block_length
                vector_index=block_start+local_vector-1
                excluded!==nothing&&excluded[vector_index]&&continue
                stored_norm=if metric===:cosine
                    resolved=vector_norms===nothing||isempty(vector_norms) ? column_norm(vectors,vector_index) : Float32(vector_norms[vector_index])
                    iszero(resolved)&&throw(ArgumentError("stored vector cannot be zero"))
                    resolved
                else
                    1.0f0
                end

                for local_query in 1:query_length
                    query_index=query_start+local_query-1
                    score=block_scores[local_vector,local_query]
                    metric===:cosine&&(score/=workspace.query_norms[query_index]*stored_norm)
                    add_top_candidate!(workspace.query_workspaces[query_index],vector_index,score,vector_index,k)
                end
            end
        end
    end

    return[top_candidate_results(metadata,workspace.query_workspaces[query_index]) for query_index in 1:query_count]
end

function score_ivf_candidates(vectors::AbstractMatrix,metadata::AbstractVector,query::AbstractVector,candidate_indices;k::Int,metric::Symbol,vector_norms::Union{Nothing,AbstractVector}=nothing,excluded::Union{Nothing,BitVector}=nothing,workspace::SearchWorkspace=search_workspace(),)
    k>0||throw(ArgumentError("k must be positive"))
    metric in (:cosine,:dot)||throw(ArgumentError("metric must be cosine or dot"))
    validate_excluded(excluded,size(vectors,2))
    vector_norms!==nothing&&!isempty(vector_norms)&&length(vector_norms)!=size(vectors,2)&&throw(DimensionMismatch("vector norm count doesnt match vectors"))
    reset_top_candidates!(workspace,k)

    if candidate_indices isa AbstractUnitRange&&eltype(vectors)===Float32&&length(candidate_indices)>=64
        score_contiguous_candidates!(workspace,vectors,query,candidate_indices;k=k,metric=metric,vector_norms=vector_norms,excluded=excluded,)
    else
        score_candidate_stream!(workspace,vectors,query,candidate_indices;k=k,metric=metric,vector_norms=vector_norms,excluded=excluded,)
    end

    return top_candidate_results(metadata,workspace)
end

function rank_scored_candidates(metadata::AbstractVector,candidate_indices::AbstractVector{<:Integer},scores::AbstractVector{<:Real};k::Int,)
    k>0||throw(ArgumentError("k must be positive"))
    length(candidate_indices)==length(scores)||throw(DimensionMismatch("candidate and score counts dont match"))
    isempty(candidate_indices)&&return NamedTuple[]

    ranked=top_k(scores,min(k,length(scores)))

    return[(
        index=Int(candidate_indices[result.index]),
        score=result.score,
        metadata=metadata[candidate_indices[result.index]],
    ) for result in ranked]
end

function validate_ivf_search(index::IVFIndex,vectors::AbstractMatrix,metadata::AbstractVector,query::AbstractVector)
    dim,vector_count=size(vectors)
    centroid_dim,_=size(index.centroids)

    length(query)==dim||throw(DimensionMismatch("query dim doesnt match vectors"))
    centroid_dim==dim||throw(DimensionMismatch("centroid dim doesnt match vectors"))
    length(metadata)==vector_count||throw(DimensionMismatch("metadata count doesnt match vectors"))

    return nothing
end

function search_ivf(index::IVFIndex,vectors::AbstractMatrix,metadata::AbstractVector,query::AbstractVector;k::Int=10,nprobe::Int=1,metric::Symbol=:cosine,excluded::Union{Nothing,BitVector}=nothing,)
    validate_ivf_search(index,vectors,metadata,query)
    workspace=search_workspace()
    selected_lists=rank_ivf_lists!(workspace.selected_lists,index,query,nprobe,workspace)
    candidate_indices=collect_ivf_candidates!(workspace.candidate_indices,index,selected_lists)

    return score_ivf_candidates(vectors,metadata,query,candidate_indices;k=k,metric=metric,vector_norms=index.vector_norms,excluded=excluded,workspace=workspace,)
end

function search_ivf_prefilter(index::IVFIndex,vectors::AbstractMatrix,metadata::AbstractVector,query::AbstractVector;k::Int=10,nprobe::Int=1,metric::Symbol=:cosine,filter::Union{NamedTuple,FilterExpr},filter_index::Union{Nothing,BitsetIndex}=nothing,excluded::Union{Nothing,BitVector}=nothing,)
    validate_ivf_search(index,vectors,metadata,query)
    normalized_filter=normalize_filter(filter)
    workspace=search_workspace()
    selected_lists=rank_ivf_lists!(workspace.selected_lists,index,query,nprobe,workspace)
    visited_indices=collect_ivf_candidates!(workspace.candidate_indices,index,selected_lists)
    candidate_indices=filter_ivf_candidates!(workspace.filtered_indices,visited_indices,metadata,normalized_filter;filter_index=filter_index,)

    return score_ivf_candidates(vectors,metadata,query,candidate_indices;k=k,metric=metric,vector_norms=index.vector_norms,excluded=excluded,workspace=workspace,)
end

function resolve_postfilter_oversample(minimum_oversample::Int,selectivity::Float64;candidate_multiplier::Float64=4.0,)
    minimum_oversample>0||throw(ArgumentError("minimum oversample must be positive"))
    0.0<=selectivity<=1.0||throw(ArgumentError("selectivity must be between 0 and 1"))
    candidate_multiplier>0||throw(ArgumentError("candidate multiplier must be positive"))
    selectivity==0&&return minimum_oversample

    return max(minimum_oversample,ceil(Int,candidate_multiplier/selectivity))
end

function search_ivf_postfilter(index::IVFIndex,vectors::AbstractMatrix,metadata::AbstractVector,query::AbstractVector;k::Int=10,nprobe::Int=1,metric::Symbol=:cosine,filter::Union{NamedTuple,FilterExpr},oversample::Int=10,excluded::Union{Nothing,BitVector}=nothing,)
    oversample>0||throw(ArgumentError("oversample must be positive"))
    normalized_filter=normalize_filter(filter)

    results=search_ivf(index,vectors,metadata,query;k=k*oversample,nprobe=nprobe,metric=metric,excluded=excluded,)
    filter!(result->matches_filter(result.metadata,normalized_filter),results)
    resize!(results,min(k,length(results)))
    return results
end
