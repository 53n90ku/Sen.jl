struct BenchmarkContext
    index::FilterAwareIVFIndex
    filter_index::BitsetIndex
end

function build_benchmark_context(ivf::IVFIndex,metadata::AbstractVector)
    index=build_filter_aware_ivf(ivf,metadata)
    filter_index=build_bitset_index(metadata)
    return BenchmarkContext(index,filter_index)
end

function build_benchmark_context(vectors::AbstractMatrix,metadata::AbstractVector;nlists::Int,iterations::Int=20,seed::Int=42,metric::Symbol=:cosine,restarts::Int=1,training_count::Int=size(vectors,2),)
    ivf=build_ivf(vectors;nlists=nlists,iterations=iterations,seed=seed,metric=metric,restarts=restarts,training_count=training_count,)
    return build_benchmark_context(ivf,metadata)
end

function benchmark_summary(recalls::Vector{Float64},result_counts::Vector{Int},visited_counts::Vector{Int},scored_counts::Vector{Int},probe_counts::Vector{Int},times_ns::Vector{Int})
    return(
        average_recall=sum(recalls)/length(recalls),
        average_results=sum(result_counts)/length(result_counts),
        average_candidates_visited=sum(visited_counts)/length(visited_counts),
        average_candidates_scored=sum(scored_counts)/length(scored_counts),
        average_lists_probed=sum(probe_counts)/length(probe_counts),
        latency=latency_summary(times_ns),
    )
end

function benchmark_summary(recalls::Vector{Float64},result_counts::Vector{Int},candidate_counts::Vector{Int},times_ns::Vector{Int})
    probe_counts=zeros(Int,length(candidate_counts))
    return benchmark_summary(recalls,result_counts,candidate_counts,candidate_counts,probe_counts,times_ns)
end

function run_benchmark(context::BenchmarkContext,vectors::AbstractMatrix,metadata::AbstractVector,queries::AbstractMatrix,filters::AbstractVector;k::Int=10,nprobe::Int=1,repetitions::Int=3,metric::Symbol=:cosine,bound_minimum_nprobe::Int=1,bound_max_nprobe::Int=nprobe,postfilter_oversample::Int=10,adaptive_postfilter::Bool=true,adaptive::Bool=false,max_nprobe::Int=nprobe,candidate_multiplier::Float64=4.0,vector_weight::Float64=0.5,filter_weight::Float64=0.5,rerank_factor::Int=4,)
    vector_dim,vector_count=size(vectors)
    query_dim,query_count=size(queries)

    vector_dim==query_dim||throw(DimensionMismatch("query dim doesnt match vectors"))
    length(metadata)==vector_count||throw(DimensionMismatch("metadata count doesnt match vectors"))
    length(filters)==query_count||throw(DimensionMismatch("filter count doesnt match queries"))
    context.filter_index.count==vector_count||throw(DimensionMismatch("benchmark context doesnt match vectors"))
    context.index.ivf.metric===metric||throw(ArgumentError("benchmark metric doesnt match index metric"))
    query_count>0||throw(ArgumentError("queries cannot be empty"))
    bound_enabled=metric===:cosine

    exact_recalls=Float64[]
    prefilter_recalls=Float64[]
    postfilter_recalls=Float64[]
    aware_recalls=Float64[]
    bound_recalls=Float64[]

    exact_counts=Int[]
    prefilter_counts=Int[]
    postfilter_counts=Int[]
    aware_counts=Int[]
    bound_counts=Int[]

    exact_visited=Int[]
    prefilter_visited=Int[]
    postfilter_visited=Int[]
    aware_visited=Int[]
    bound_visited=Int[]

    exact_scored=Int[]
    prefilter_scored=Int[]
    postfilter_scored=Int[]
    aware_scored=Int[]
    bound_scored=Int[]

    exact_probes=Int[]
    prefilter_probes=Int[]
    postfilter_probes=Int[]
    aware_probes=Int[]
    bound_probes=Int[]

    exact_times=Int[]
    prefilter_times=Int[]
    postfilter_times=Int[]
    aware_times=Int[]
    bound_times=Int[]
    selectivities=Float64[]

    for query_index in 1:query_count
        query=@view queries[:,query_index]
        filter=filters[query_index]
        selectivity=estimate_selectivity(context.filter_index,filter)
        resolved_oversample=adaptive_postfilter ? resolve_postfilter_oversample(postfilter_oversample,selectivity;candidate_multiplier=candidate_multiplier,) : postfilter_oversample

        exact_measurement=measure_latency(
            ()->search_exact(vectors,metadata,query;k=k,metric=metric,filter=filter,filter_index=context.filter_index,vector_norms=context.index.ivf.vector_norms,);
            repetitions=repetitions,
        )
        exact_results=exact_measurement.result
        truth_ids=Int[result.index for result in exact_results]

        prefilter_measurement=measure_latency(
            ()->search_ivf_prefilter(context.index,vectors,metadata,query;k=k,nprobe=nprobe,metric=metric,filter=filter,);
            repetitions=repetitions,
        )
        prefilter_results=prefilter_measurement.result
        prefilter_ids=Int[result.index for result in prefilter_results]

        postfilter_measurement=measure_latency(
            ()->search_ivf_postfilter(context.index.ivf,vectors,metadata,query;k=k,nprobe=nprobe,metric=metric,filter=filter,oversample=resolved_oversample,);
            repetitions=repetitions,
        )
        postfilter_results=postfilter_measurement.result
        postfilter_ids=Int[result.index for result in postfilter_results]

        aware_measurement=measure_latency(
            ()->search_filter_aware_ivf(context.index,vectors,metadata,query;k=k,nprobe=nprobe,metric=metric,filter=filter,adaptive=adaptive,max_nprobe=max_nprobe,candidate_multiplier=candidate_multiplier,vector_weight=vector_weight,filter_weight=filter_weight,rerank_factor=rerank_factor,);
            repetitions=repetitions,
        )
        aware_results=aware_measurement.result
        aware_ids=Int[result.index for result in aware_results]

        if bound_enabled
            bound_measurement=measure_latency(
                ()->search_filter_aware_bound_with_stats(context.index,vectors,metadata,query;filter=filter,k=k,minimum_nprobe=bound_minimum_nprobe,max_nprobe=bound_max_nprobe,metric=metric,);
                repetitions=repetitions,
            )
            bound_stats=bound_measurement.result
            bound_results=bound_stats.results
            bound_ids=Int[result.index for result in bound_results]

            push!(bound_recalls,recall_at_k(bound_ids,truth_ids,k))
            push!(bound_counts,length(bound_results))
            push!(bound_visited,bound_stats.visited)
            push!(bound_scored,bound_stats.scored)
            push!(bound_probes,bound_stats.probed_lists)
            append!(bound_times,bound_measurement.times_ns)
        end

        exact_work=count_exact_candidates(context.filter_index,filter)
        prefilter_work=count_ivf_prefilter_work(context.index,metadata,query,filter;nprobe=nprobe,)
        postfilter_work=count_ivf_candidates(context.index.ivf,query;nprobe=nprobe,)
        aware_work=count_filter_aware_work(context.index,metadata,query,filter;k=k,nprobe=nprobe,adaptive=adaptive,max_nprobe=max_nprobe,candidate_multiplier=candidate_multiplier,vector_weight=vector_weight,filter_weight=filter_weight,rerank_factor=rerank_factor,)

        push!(exact_recalls,1.0)
        push!(prefilter_recalls,recall_at_k(prefilter_ids,truth_ids,k))
        push!(postfilter_recalls,recall_at_k(postfilter_ids,truth_ids,k))
        push!(aware_recalls,recall_at_k(aware_ids,truth_ids,k))

        push!(exact_counts,length(exact_results))
        push!(prefilter_counts,length(prefilter_results))
        push!(postfilter_counts,length(postfilter_results))
        push!(aware_counts,length(aware_results))

        push!(exact_visited,exact_work)
        push!(prefilter_visited,prefilter_work.visited)
        push!(postfilter_visited,postfilter_work)
        push!(aware_visited,aware_work.visited)

        push!(exact_scored,exact_work)
        push!(prefilter_scored,prefilter_work.scored)
        push!(postfilter_scored,postfilter_work)
        push!(aware_scored,aware_work.scored)

        push!(exact_probes,0)
        push!(prefilter_probes,nprobe)
        push!(postfilter_probes,nprobe)
        push!(aware_probes,aware_work.probed_lists)

        append!(exact_times,exact_measurement.times_ns)
        append!(prefilter_times,prefilter_measurement.times_ns)
        append!(postfilter_times,postfilter_measurement.times_ns)
        append!(aware_times,aware_measurement.times_ns)
        push!(selectivities,selectivity)
    end

    return(
        exact=benchmark_summary(exact_recalls,exact_counts,exact_visited,exact_scored,exact_probes,exact_times),
        ivf_prefilter=benchmark_summary(prefilter_recalls,prefilter_counts,prefilter_visited,prefilter_scored,prefilter_probes,prefilter_times),
        ivf_postfilter=benchmark_summary(postfilter_recalls,postfilter_counts,postfilter_visited,postfilter_scored,postfilter_probes,postfilter_times),
        filter_aware=benchmark_summary(aware_recalls,aware_counts,aware_visited,aware_scored,aware_probes,aware_times),
        filter_aware_bound=bound_enabled ? benchmark_summary(bound_recalls,bound_counts,bound_visited,bound_scored,bound_probes,bound_times) : nothing,
        average_selectivity=sum(selectivities)/length(selectivities),
    )
end

function run_benchmark(vectors::AbstractMatrix,metadata::AbstractVector,queries::AbstractMatrix,filters::AbstractVector;nlists::Int,k::Int=10,nprobe::Int=1,iterations::Int=20,seed::Int=42,repetitions::Int=3,metric::Symbol=:cosine,restarts::Int=1,training_count::Int=size(vectors,2),bound_minimum_nprobe::Int=1,bound_max_nprobe::Int=nprobe,postfilter_oversample::Int=10,adaptive_postfilter::Bool=true,adaptive::Bool=false,max_nprobe::Int=nprobe,candidate_multiplier::Float64=4.0,vector_weight::Float64=0.5,filter_weight::Float64=0.5,rerank_factor::Int=4,)
    context=build_benchmark_context(vectors,metadata;nlists=nlists,iterations=iterations,seed=seed,metric=metric,restarts=restarts,training_count=training_count,)

    return run_benchmark(
        context,
        vectors,
        metadata,
        queries,
        filters;
        k=k,
        nprobe=nprobe,
        repetitions=repetitions,
        metric=metric,
        bound_minimum_nprobe=bound_minimum_nprobe,
        bound_max_nprobe=bound_max_nprobe,
        postfilter_oversample=postfilter_oversample,
        adaptive_postfilter=adaptive_postfilter,
        adaptive=adaptive,
        max_nprobe=max_nprobe,
        candidate_multiplier=candidate_multiplier,
        vector_weight=vector_weight,
        filter_weight=filter_weight,
        rerank_factor=rerank_factor,
    )
end
