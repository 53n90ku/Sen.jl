function recall_at_k(predicted_ids::AbstractVector{<:Integer},truth_ids::AbstractVector{<:Integer},k::Int,)::Float64
    k>0||throw(ArgumentError("k must be positive"))

    predicted_top=predicted_ids[1:min(k,length(predicted_ids))]
    truth_top=truth_ids[1:min(k,length(truth_ids))]

    if isempty(truth_top)
        return isempty(predicted_top) ? 1.0 : 0.0
    end

    matches=length(intersect(Set(predicted_top),Set(truth_top)))
    return matches/length(truth_top)
end

function latency_summary(times_ns::AbstractVector{<:Integer})
    isempty(times_ns)&&throw(ArgumentError("latency measurements cannot be empty"))
    any(time->time<0,times_ns)&&throw(ArgumentError("latency measurements cannot be negative"))

    times_ms=sort(Float64.(times_ns)./1_000_000)
    count=length(times_ms)
    p50_index=clamp(ceil(Int,0.50*count),1,count)
    p95_index=clamp(ceil(Int,0.95*count),1,count)

    return(
        minimum_ms=first(times_ms),
        mean_ms=sum(times_ms)/count,
        p50_ms=times_ms[p50_index],
        p95_ms=times_ms[p95_index],
        maximum_ms=last(times_ms),
    )
end

function measure_latency(search_function::Function;repetitions::Int=10,)
    repetitions>0||throw(ArgumentError("repetitions must be positive"))

    search_function()
    times_ns=Vector{Int}(undef,repetitions)
    latest_result=nothing

    for repetition in 1:repetitions
        start_time=time_ns()
        latest_result=search_function()
        times_ns[repetition]=time_ns()-start_time
    end

    return(result=latest_result,times_ns=times_ns,summary=latency_summary(times_ns),)
end

function count_exact_candidates(index::BitsetIndex,filter::Union{Nothing,NamedTuple,FilterExpr})
    filter===nothing&&return index.count
    return count(identity,evaluate_filter(index,filter))
end

function count_ivf_candidates(index::IVFIndex,query::AbstractVector;nprobe::Int,)
    selected_lists=rank_ivf_lists(index,query;nprobe=nprobe,)
    return sum(length(index.lists[list_index]) for list_index in selected_lists)
end

function count_ivf_prefilter_work(index::IVFIndex,metadata::AbstractVector,query::AbstractVector,filter::Union{NamedTuple,FilterExpr};nprobe::Int,filter_index::Union{Nothing,BitsetIndex}=nothing,)
    selected_lists=rank_ivf_lists(index,query;nprobe=nprobe,)
    visited_indices=collect_ivf_candidates(index,selected_lists)
    candidate_indices=filter_ivf_candidates(visited_indices,metadata,filter;filter_index=filter_index,)

    return(visited=length(visited_indices),scored=length(candidate_indices),probed_lists=length(selected_lists),)
end

function count_ivf_prefilter_work(index::FilterAwareIVFIndex,metadata::AbstractVector,query::AbstractVector,filter::Union{NamedTuple,FilterExpr};nprobe::Int,)
    length(metadata)==sum(length,index.ivf.lists)||throw(DimensionMismatch("metadata count doesnt match index"))
    selected_lists=rank_ivf_lists(index.ivf,query;nprobe=nprobe,)
    candidate_indices=collect_filtered_list_candidates(index,selected_lists,filter)

    return(visited=length(candidate_indices),scored=length(candidate_indices),probed_lists=length(selected_lists),)
end

function count_filter_aware_work(index::FilterAwareIVFIndex,metadata::AbstractVector,query::AbstractVector,filter::Union{NamedTuple,FilterExpr};k::Int=10,nprobe::Int,adaptive::Bool=false,max_nprobe::Int=nprobe,candidate_multiplier::Float64=4.0,vector_weight::Float64=0.5,filter_weight::Float64=0.5,rerank_factor::Int=4,)
    selection=select_filter_aware_candidates(
        index,
        metadata,
        query,
        filter;
        k=k,
        nprobe=nprobe,
        adaptive=adaptive,
        max_nprobe=max_nprobe,
        candidate_multiplier=candidate_multiplier,
        vector_weight=vector_weight,
        filter_weight=filter_weight,
        rerank_factor=rerank_factor,
    )

    return(
        visited=length(selection.visited_indices),
        scored=length(selection.candidate_indices),
        probed_lists=length(selection.selected_lists),
    )
end

function count_filter_aware_candidates(index::FilterAwareIVFIndex,metadata::AbstractVector,query::AbstractVector,filter::Union{NamedTuple,FilterExpr};k::Int=10,nprobe::Int,adaptive::Bool=false,max_nprobe::Int=nprobe,candidate_multiplier::Float64=4.0,vector_weight::Float64=0.5,filter_weight::Float64=0.5,rerank_factor::Int=4,)
    work=count_filter_aware_work(
        index,
        metadata,
        query,
        filter;
        k=k,
        nprobe=nprobe,
        adaptive=adaptive,
        max_nprobe=max_nprobe,
        candidate_multiplier=candidate_multiplier,
        vector_weight=vector_weight,
        filter_weight=filter_weight,
        rerank_factor=rerank_factor,
    )

    return work.scored
end
