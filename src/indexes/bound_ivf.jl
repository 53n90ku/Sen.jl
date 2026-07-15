function rank_bound_lists(index::FilterAwareIVFIndex,query::AbstractVector,filter::NamedTuple,)
    bounds=list_score_upper_bounds(index.ivf,query)
    counts=[estimate_list_filter_count(index,list_index,filter) for list_index in eachindex(index.ivf.lists)]
    lists=Int[list_index for list_index in eachindex(index.ivf.lists) if counts[list_index]>0]
    sort!(lists;by=list_index->(bounds[list_index],counts[list_index]),rev=true,)

    return(lists=lists,bounds=bounds,counts=counts,)
end

function suffix_maximums(values::AbstractVector{<:Real})
    suffix=fill(-Inf32,length(values)+1)

    for index in length(values):-1:1
        suffix[index]=max(Float32(values[index]),suffix[index+1])
    end

    return suffix
end

function add_top_candidate!(indices::Vector{Int},scores::Vector{Float32},index::Int,score::Float32,k::Int,minimum_score::Float32,minimum_position::Int)
    if length(scores)<k
        push!(indices,index)
        push!(scores,score)
        return length(scores)==k ? findmin(scores) : (minimum_score,minimum_position)
    end

    if score>minimum_score
        indices[minimum_position]=index
        scores[minimum_position]=score
        return findmin(scores)
    end

    return(minimum_score,minimum_position)
end

function search_filter_aware_bound_with_stats(index::FilterAwareIVFIndex,vectors::AbstractMatrix,metadata::AbstractVector,query::AbstractVector;filter::NamedTuple,k::Int=10,minimum_nprobe::Int=1,max_nprobe::Int=length(index.ivf.lists),metric::Symbol=:cosine,excluded::Union{Nothing,BitVector}=nothing,)
    validate_ivf_search(index.ivf,vectors,metadata,query)
    validate_excluded(excluded,size(vectors,2))
    k>0||throw(ArgumentError("k must be positive"))
    metric===:cosine||throw(ArgumentError("bound search requires cosine similarity"))
    index.ivf.metric===:cosine||throw(ArgumentError("bound search requires a cosine index"))
    1<=minimum_nprobe<=length(index.ivf.lists)||throw(ArgumentError("minimum nprobe must be between 1 and list count"))
    minimum_nprobe<=max_nprobe<=length(index.ivf.lists)||throw(ArgumentError("max nprobe must be between minimum nprobe and list count"))

    ranking=rank_bound_lists(index,query,filter)
    isempty(ranking.lists)&&return(
        results=NamedTuple[],
        visited=0,
        scored=0,
        probed_lists=0,
        stopped_by_bound=false,
        exact=true,
    )

    ordered_bounds=Float32[ranking.bounds[list_index] for list_index in ranking.lists]
    remaining_bounds=suffix_maximums(ordered_bounds)
    query_norm=vector_norm(query)
    top_indices=Int[]
    top_scores=Float32[]
    minimum_score=-Inf32
    minimum_position=0
    visited=0
    scored=0
    probed_lists=0
    stopped_by_bound=false

    for(position,list_index) in enumerate(ranking.lists)
        probed_lists>=max_nprobe&&break
        candidates=filtered_list_candidates(index,list_index,filter)
        probed_lists+=1

        for vector_index in candidates
            excluded!==nothing&&excluded[vector_index]&&continue
            visited+=1
            stored_norm=index.ivf.vector_norms[vector_index]
            iszero(stored_norm)&&throw(ArgumentError("stored vector cannot be zero"))
            score=column_dot(query,vectors,vector_index)/(query_norm*stored_norm)
            scored+=1
            minimum_score,minimum_position=add_top_candidate!(top_indices,top_scores,vector_index,score,k,minimum_score,minimum_position)
        end

        if probed_lists>=minimum_nprobe&&length(top_scores)==k
            threshold=minimum_score
            maximum_unvisited=remaining_bounds[position+1]

            if maximum_unvisited<threshold
                stopped_by_bound=true
                break
            end
        end
    end

    results=rank_scored_candidates(metadata,top_indices,top_scores;k=k,)
    exact=stopped_by_bound||probed_lists==length(ranking.lists)

    return(
        results=results,
        visited=visited,
        scored=scored,
        probed_lists=probed_lists,
        stopped_by_bound=stopped_by_bound,
        exact=exact,
    )
end

function search_filter_aware_bound(index::FilterAwareIVFIndex,vectors::AbstractMatrix,metadata::AbstractVector,query::AbstractVector;filter::NamedTuple,k::Int=10,minimum_nprobe::Int=1,max_nprobe::Int=length(index.ivf.lists),metric::Symbol=:cosine,excluded::Union{Nothing,BitVector}=nothing,)
    stats=search_filter_aware_bound_with_stats(index,vectors,metadata,query;filter=filter,k=k,minimum_nprobe=minimum_nprobe,max_nprobe=max_nprobe,metric=metric,excluded=excluded,)
    return stats.results
end
