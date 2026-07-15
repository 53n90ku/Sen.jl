const SEARCH_SCORE_BLOCK_SIZE=256
const SEARCH_QUERY_BLOCK_SIZE=16
const SEARCH_WORKSPACE_KEY=:sen_search_workspace
const BATCH_SEARCH_WORKSPACE_KEY=:sen_batch_search_workspace

mutable struct SearchWorkspace
    selected_lists::Vector{Int}
    candidate_indices::Vector{Int}
    filtered_indices::Vector{Int}
    heap_indices::Vector{Int}
    heap_orders::Vector{Int}
    heap_scores::Vector{Float32}
    centroid_distances::Vector{Float32}
    block_scores::Vector{Float32}
    query_buffer::Vector{Float32}
end

function SearchWorkspace()
    return SearchWorkspace(Int[],Int[],Int[],Int[],Int[],Float32[],Float32[],Float32[],Float32[])
end

mutable struct BatchSearchWorkspace
    score_block::Matrix{Float32}
    query_buffer::Matrix{Float32}
    query_norms::Vector{Float32}
    query_workspaces::Vector{SearchWorkspace}
end

function BatchSearchWorkspace()
    return BatchSearchWorkspace(Matrix{Float32}(undef,0,0),Matrix{Float32}(undef,0,0),Float32[],SearchWorkspace[])
end

function search_workspace()
    storage=task_local_storage()
    return get!(storage,SEARCH_WORKSPACE_KEY) do
        SearchWorkspace()
    end
end

function batch_search_workspace()
    storage=task_local_storage()
    return get!(storage,BATCH_SEARCH_WORKSPACE_KEY) do
        BatchSearchWorkspace()
    end
end

@inline function top_candidate_better(left_score::Float32,left_order::Int,right_score::Float32,right_order::Int)
    return isless(right_score,left_score)||(isequal(left_score,right_score)&&left_order<right_order)
end

function reset_top_candidates!(workspace::SearchWorkspace,k::Int)
    empty!(workspace.heap_indices)
    empty!(workspace.heap_orders)
    empty!(workspace.heap_scores)
    sizehint!(workspace.heap_indices,k)
    sizehint!(workspace.heap_orders,k)
    sizehint!(workspace.heap_scores,k)
    return workspace
end

@inline function add_top_candidate!(workspace::SearchWorkspace,index::Int,score::Float32,order::Int,k::Int)
    count=length(workspace.heap_scores)

    if count<k
        push!(workspace.heap_indices,index)
        push!(workspace.heap_orders,order)
        push!(workspace.heap_scores,score)
        count+=1
    else
        top_candidate_better(score,order,workspace.heap_scores[end],workspace.heap_orders[end])||return workspace
    end

    insertion=count

    while insertion>1&&top_candidate_better(score,order,workspace.heap_scores[insertion-1],workspace.heap_orders[insertion-1])
        workspace.heap_indices[insertion]=workspace.heap_indices[insertion-1]
        workspace.heap_orders[insertion]=workspace.heap_orders[insertion-1]
        workspace.heap_scores[insertion]=workspace.heap_scores[insertion-1]
        insertion-=1
    end

    workspace.heap_indices[insertion]=index
    workspace.heap_orders[insertion]=order
    workspace.heap_scores[insertion]=score
    return workspace
end

function sort_top_candidates!(workspace::SearchWorkspace)
    return workspace
end

function top_k(scores::AbstractVector,k::Int)
    k>0||throw(ArgumentError("k should be positive"))
    k<=length(scores)||throw(ArgumentError("k cant exceed number of scores"))
    workspace=search_workspace()
    reset_top_candidates!(workspace,k)

    for index in eachindex(scores)
        add_top_candidate!(workspace,Int(index),Float32(scores[index]),Int(index),k)
    end

    sort_top_candidates!(workspace)
    return[(index=workspace.heap_indices[position],score=workspace.heap_scores[position],) for position in eachindex(workspace.heap_scores)]
end
