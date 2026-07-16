function aggregate_probe_curve(rows::AbstractVector)
    isempty(rows)&&throw(ArgumentError("probe curve cannot be empty"))
    groups=Dict{Tuple{Int,Symbol},Vector{Any}}()

    for row in rows
        push!(get!(groups, (row.nprobe, row.bucket), Any[]), row)
        push!(get!(groups, (row.nprobe, :all), Any[]), row)
    end

    aggregates=NamedTuple[]
    for ((nprobe, bucket), values) in groups
        recalls=Float64[value.truth_list_recall for value in values]
        candidates=Float64[value.candidates_visited for value in values]
        matching=Float64[value.matching_candidates for value in values]
        fractions=Float64[value.database_fraction for value in values]
        push!(
            aggregates,
            (
                nprobe = nprobe,
                bucket = bucket,
                query_count = length(values),
                recall = sum(recalls)/length(recalls),
                minimum_recall = minimum(recalls),
                p50_candidates = percentile(candidates, 0.50),
                p95_candidates = percentile(candidates, 0.95),
                average_candidates = sum(candidates)/length(candidates),
                average_matching_candidates = sum(matching)/length(matching),
                average_database_fraction = sum(fractions)/length(fractions),
            ),
        )
    end
    sort!(aggregates; by = row->(row.nprobe, string(row.bucket)))
    return aggregates
end

function ivf_sweep_configuration_hash(
    dataset_hash::AbstractString,
    query_indices::AbstractVector{<:Integer};
    nlists::Int,
    training_count::Int,
    iterations::Int,
    restarts::Int,
    seed::Int,
    metric::Symbol = :cosine,
)
    value=join(
        (
            dataset_hash,
            join(query_indices, ','),
            nlists,
            training_count,
            iterations,
            restarts,
            seed,
            metric,
        ),
        '|',
    )
    return bytes2hex(SHA.sha256(value))
end

function sweep_configuration_name(
    configuration_hash::AbstractString;
    nlists::Int,
    training_count::Int,
    iterations::Int,
    restarts::Int,
    seed::Int,
)
    return "lists-$(nlists)-train-$(training_count)-iter-$(iterations)-restart-$(restarts)-seed-$(seed)-$(configuration_hash[1:12])"
end

function sweep_checkpoint_data(
    dataset::ArxivFANNSDataset,
    configuration_hash::String,
    index_diagnostics,
    aggregate_rows;
    nlists::Int,
    training_count::Int,
    iterations::Int,
    restarts::Int,
    seed::Int,
    build_seconds::Float64,
)
    eligible=[row for row in aggregate_rows if row.bucket===:all&&row.recall>=0.97]
    selected=isempty(eligible) ?
             first(
        sort(
            [row for row in aggregate_rows if row.bucket===:all];
            by = row->(-row.recall, row.average_candidates, row.nprobe),
        ),
    ) : first(sort(eligible; by = row->(row.average_candidates, row.nprobe)))
    return Dict(
        "complete"=>true,
        "configuration_hash"=>configuration_hash,
        "dataset_hash"=>dataset.manifest.preprocessing_hash,
        "query_indices"=>dataset.train_indices,
        "nlists"=>nlists,
        "training_count"=>training_count,
        "iterations"=>iterations,
        "restarts"=>restarts,
        "seed"=>seed,
        "build_seconds"=>build_seconds,
        "selected_nprobe"=>selected.nprobe,
        "selected_recall"=>selected.recall,
        "selected_average_candidates"=>selected.average_candidates,
        "selected_database_fraction"=>selected.average_database_fraction,
        "empty_lists"=>index_diagnostics.empty_lists,
        "list_size_cv"=>index_diagnostics.list_size_cv,
        "list_size_gini"=>index_diagnostics.list_size_gini,
        "maximum_to_average_list_ratio"=>index_diagnostics.maximum_to_average_list_ratio,
        "mean_quantization_loss"=>index_diagnostics.mean_quantization_loss,
        "p95_quantization_loss"=>index_diagnostics.p95_quantization_loss,
        "maximum_centroid_similarity"=>index_diagnostics.maximum_centroid_similarity,
        "near_duplicate_centroids"=>index_diagnostics.near_duplicate_centroids,
    )
end

function run_ivf_quality_configuration(
    dataset::ArxivFANNSDataset,
    output_path::AbstractString;
    nlists::Int,
    training_count::Int,
    iterations::Int = 8,
    restarts::Int = 2,
    seed::Int = dataset.manifest.sampling_seed,
    force::Bool = false,
)
    nlists<=training_count<=size(dataset.vectors, 2)||throw(
        ArgumentError("training count must be between nlists and vector count"),
    )
    configuration_hash=ivf_sweep_configuration_hash(
        dataset.manifest.preprocessing_hash,
        dataset.train_indices;
        nlists = nlists,
        training_count = training_count,
        iterations = iterations,
        restarts = restarts,
        seed = seed,
    )
    name=sweep_configuration_name(
        configuration_hash;
        nlists = nlists,
        training_count = training_count,
        iterations = iterations,
        restarts = restarts,
        seed = seed,
    )
    path=joinpath(output_path, name)
    checkpoint_path=joinpath(path, "checkpoint.toml")
    index_path=joinpath(path, "index")

    if isfile(checkpoint_path)&&isfile(joinpath(index_path, "index.bin"))&&!force
        checkpoint=TOML.parsefile(checkpoint_path)
        get(checkpoint, "complete", false)||throw(
            ArgumentError("incomplete IVF sweep checkpoint exists at $(path)"),
        )
        String(checkpoint["configuration_hash"])==configuration_hash||throw(
            ArgumentError("IVF sweep checkpoint configuration changed"),
        )
        return (status = :cached, path = path, checkpoint = checkpoint)
    end

    mkpath(path)
    index=nothing
    build_seconds=@elapsed index=build_ivf(
        dataset.vectors;
        nlists = nlists,
        iterations = iterations,
        restarts = restarts,
        training_count = training_count,
        seed = seed,
        metric = :cosine,
    )
    index_diagnostics=ivf_index_diagnostics(index, dataset.vectors)
    nprobes=probe_schedule(nlists)
    routing_rows=ivf_routing_diagnostics(
        index,
        dataset.train_queries,
        dataset.train_truth;
        metadata = dataset.metadata,
        filters = dataset.train_filters,
    )
    curve_rows=ivf_probe_curve(
        index,
        dataset.train_queries,
        dataset.train_truth,
        nprobes;
        metadata = dataset.metadata,
        filters = dataset.train_filters,
    )
    aggregate_rows=aggregate_probe_curve(curve_rows)
    write_tsv(joinpath(path, "routing.tsv"), routing_rows)
    write_tsv(joinpath(path, "probe_curve.tsv"), curve_rows)
    write_tsv(joinpath(path, "aggregate.tsv"), aggregate_rows)
    write_tsv(joinpath(path, "index_diagnostics.tsv"), [index_diagnostics])
    save_ivf_index(index_path, index)
    checkpoint=sweep_checkpoint_data(
        dataset,
        configuration_hash,
        index_diagnostics,
        aggregate_rows;
        nlists = nlists,
        training_count = training_count,
        iterations = iterations,
        restarts = restarts,
        seed = seed,
        build_seconds = build_seconds,
    )
    temporary="$(checkpoint_path).tmp"
    open(temporary, "w") do io
        TOML.print(io, checkpoint)
    end
    mv(temporary, checkpoint_path; force = true)
    return (status = :complete, path = path, checkpoint = checkpoint)
end

function run_ivf_quality_sweep(
    dataset::ArxivFANNSDataset,
    output_path::AbstractString;
    nlists_values::AbstractVector{<:Integer} = [64, 128, 256, 512],
    training_count::Int = 20_000,
    iterations::Int = 8,
    restarts::Int = 2,
    seed::Int = dataset.manifest.sampling_seed,
    force::Bool = false,
)
    mkpath(output_path)
    results=NamedTuple[]
    for nlists in nlists_values
        println(
            "IVF sweep nlists=$(nlists) training=$(training_count) iterations=$(iterations) restarts=$(restarts)",
        )
        result=run_ivf_quality_configuration(
            dataset,
            output_path;
            nlists = Int(nlists),
            training_count = training_count,
            iterations = iterations,
            restarts = restarts,
            seed = seed,
            force = force,
        )
        println("IVF sweep status=$(result.status) path=$(result.path)")
        push!(results, result)
    end
    return results
end

function evaluation_from_rows(rows::AbstractVector)
    isempty(rows)&&throw(ArgumentError("evaluation rows cannot be empty"))
    grouped=Dict{Int,Any}()
    for row in rows
        grouped[row.query_index]=row
    end
    query_rows=collect(values(grouped))
    recalls=Float64[row.recall for row in query_rows]
    result_counts=Int[row.result_count for row in query_rows]
    visited=Int[row.candidates_visited for row in query_rows]
    scored=Int[row.candidates_scored for row in query_rows]
    probes=Int[row.lists_probed for row in query_rows]
    times_ns=Int[round(Int, row.latency_ms*1_000_000) for row in rows]
    return benchmark_summary(recalls, result_counts, visited, scored, probes, times_ns)
end

function bootstrap_recall_interval(
    rows::AbstractVector;
    samples::Int = 2_000,
    seed::Int = 42,
    confidence::Float64 = 0.95,
)
    samples>0||throw(ArgumentError("bootstrap samples must be positive"))
    0<confidence<1||throw(ArgumentError("confidence must be between zero and one"))
    grouped=Dict{Int,Float64}()
    for row in rows
        grouped[row.query_index]=Float64(row.recall)
    end
    recalls=collect(values(grouped))
    isempty(recalls)&&throw(ArgumentError("bootstrap rows cannot be empty"))
    rng=MersenneTwister(seed)
    estimates=Vector{Float64}(undef, samples)
    for sample = 1:samples
        estimates[sample]=sum(
            recalls[rand(rng, 1:length(recalls))] for _ in eachindex(recalls)
        )/length(recalls)
    end
    alpha=(1-confidence)/2
    return (
        lower = percentile(estimates, alpha),
        estimate = sum(recalls)/length(recalls),
        upper = percentile(estimates, 1-alpha),
        confidence = confidence,
        samples = samples,
    )
end

function evaluate_hybrid_validation(
    dataset::ArxivFANNSDataset,
    index::IVFIndex;
    nprobe::Int,
    k::Int = 10,
    repetitions::Int = 3,
    rare_threshold::Float64 = 0.01,
    target_recall::Float64 = 0.95,
    split::Symbol = :validation,
)
    0<=rare_threshold<=1||throw(
        ArgumentError("rare threshold must be between zero and one"),
    )
    1<=nprobe<=length(index.lists)||throw(
        ArgumentError("nprobe must be between one and list count"),
    )
    spec=real_experiment_spec(
        dataset;
        name = "ivf-hybrid-validation",
        nlists = length(index.lists),
        nprobes = [nprobe],
        k = k,
        repetitions = repetitions,
        target_recall = target_recall,
        training_count = size(dataset.vectors, 2),
        iterations = 1,
        restarts = 1,
    )
    context=build_benchmark_context(index, dataset.metadata)
    exact=evaluate_benchmark_method(
        spec,
        context,
        dataset.vectors,
        dataset.metadata,
        dataset.heldout_queries,
        dataset.heldout_filters,
        dataset.heldout_truth,
        :exact,
        0;
        split = split,
    )
    approximate=evaluate_benchmark_method(
        spec,
        context,
        dataset.vectors,
        dataset.metadata,
        dataset.heldout_queries,
        dataset.heldout_filters,
        dataset.heldout_truth,
        :ivf_prefilter,
        nprobe;
        split = split,
    )
    selectivities=dataset_selectivities(dataset.metadata, dataset.heldout_filters)
    exact_rows=Dict((row.query_index, row.repetition)=>row for row in exact.rows)
    approximate_rows=Dict(
        (row.query_index, row.repetition)=>row for row in approximate.rows
    )
    hybrid_rows=NamedTuple[]

    for query_index in eachindex(selectivities)
        selected_method=selectivities[query_index]<=rare_threshold ? :exact : :ivf_prefilter
        source=selected_method===:exact ? exact_rows : approximate_rows
        for repetition = 1:repetitions
            row=source[(query_index, repetition)]
            push!(
                hybrid_rows,
                merge(
                    row,
                    (
                        method = "hybrid",
                        selected_method = String(selected_method),
                        selectivity = selectivities[query_index],
                        bucket = String(selectivity_bucket(selectivities[query_index])),
                    ),
                ),
            )
        end
    end

    hybrid_summary=evaluation_from_rows(hybrid_rows)
    bucket_summaries=Dict{Symbol,Any}()
    for bucket in (:rare, :medium, :broad)
        bucket_rows=[row for row in hybrid_rows if row.bucket==String(bucket)]
        isempty(bucket_rows)||setindex!(
            bucket_summaries,
            evaluation_from_rows(bucket_rows),
            bucket,
        )
    end
    hybrid_p95=hybrid_summary.latency.p95_ms
    return (
        nprobe = nprobe,
        rare_threshold = rare_threshold,
        exact = exact,
        approximate = approximate,
        hybrid = (summary = hybrid_summary, rows = hybrid_rows),
        buckets = bucket_summaries,
        recall_interval = bootstrap_recall_interval(
            hybrid_rows;
            seed = dataset.manifest.sampling_seed,
        ),
        bucket_intervals = Dict(
            bucket=>bootstrap_recall_interval(
                [row for row in hybrid_rows if row.bucket==String(bucket)];
                seed = dataset.manifest.sampling_seed+index,
            ) for (index, bucket) in enumerate(keys(bucket_summaries))
        ),
        speedup_p95 = iszero(hybrid_p95) ? Inf : exact.summary.latency.p95_ms/hybrid_p95,
        recall_gate = hybrid_summary.average_recall>=target_recall,
    )
end

function save_hybrid_validation(path::AbstractString, result)
    mkpath(path)
    raw=write_tsv(joinpath(path, "raw_results.tsv"), result.hybrid.rows)
    summary_rows=NamedTuple[]
    for (method, evaluation) in (
        (:exact, result.exact),
        (:ivf_prefilter, result.approximate),
        (:hybrid, result.hybrid),
    )
        summary=evaluation.summary
        push!(
            summary_rows,
            (
                method = method,
                recall = summary.average_recall,
                p50_ms = summary.latency.p50_ms,
                p95_ms = summary.latency.p95_ms,
                candidates_visited = summary.average_candidates_visited,
                candidates_scored = summary.average_candidates_scored,
                lists_probed = summary.average_lists_probed,
            ),
        )
    end
    for (bucket, summary) in
        sort!(collect(result.buckets); by = value->string(first(value)))
        push!(
            summary_rows,
            (
                method = Symbol("hybrid_$(bucket)"),
                recall = summary.average_recall,
                p50_ms = summary.latency.p50_ms,
                p95_ms = summary.latency.p95_ms,
                candidates_visited = summary.average_candidates_visited,
                candidates_scored = summary.average_candidates_scored,
                lists_probed = summary.average_lists_probed,
            ),
        )
    end
    summary=write_tsv(joinpath(path, "summary.tsv"), summary_rows)
    return (raw = raw, summary = summary)
end
