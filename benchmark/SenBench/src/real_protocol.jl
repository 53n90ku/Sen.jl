function select_real_method(spec::ExperimentSpec, evaluations::AbstractDict, method::Symbol)
    points=[
        (
            nprobe = nprobe,
            recall = evaluation.summary.average_recall,
            p50_ms = evaluation.summary.latency.p50_ms,
            p95_ms = evaluation.summary.latency.p95_ms,
            candidates = evaluation.summary.average_candidates_scored,
        ) for (nprobe, evaluation) in evaluations
    ]
    isempty(points)&&throw(ArgumentError("real method evaluations cannot be empty"))
    sort!(points; by = point->point.nprobe)
    selection_target=min(1.0, spec.target_recall+spec.selection_margin)
    eligible=[point for point in points if point.recall>=selection_target]

    if isempty(eligible)&&selection_target>spec.target_recall
        selection_target=spec.target_recall
        eligible=[point for point in points if point.recall>=selection_target]
    end

    base=isempty(eligible) ?
         first(
        sort(
            points;
            by = point->(-point.recall, point.nprobe, point.candidates, point.p95_ms),
        ),
    ) : first(sort(eligible; by = point->(point.nprobe, point.candidates, point.p95_ms)))
    safety_nprobe=min(
        maximum(point.nprobe for point in points),
        ceil(Int, base.nprobe*spec.probe_safety_factor),
    )
    safety=[point for point in points if point.nprobe>=safety_nprobe]
    selected=first(
        sort(
            safety;
            by = point->(point.nprobe, -point.recall, point.candidates, point.p95_ms),
        ),
    )
    return (
        method = method,
        nprobe = selected.nprobe,
        train_recall = selected.recall,
        train_p50_ms = selected.p50_ms,
        train_p95_ms = selected.p95_ms,
        train_candidates = selected.candidates,
        achieved = selected.recall>=spec.target_recall,
        selection_target = selection_target,
        postfilter_candidate_multiplier = 0.0,
    )
end

function benchmark_qps(evaluation)
    latencies=Float64[row.latency_ms for row in evaluation.rows]
    isempty(latencies)&&return 0.0
    total=sum(latencies)
    return total==0 ? Inf : 1000*length(latencies)/total
end

function real_experiment_spec(
    dataset::ArxivFANNSDataset;
    name::AbstractString = "arxiv-fanns-100k",
    nlists::Int = 64,
    nprobes::AbstractVector{<:Integer} = [1, 2, 4, 8, 16, 32, 64],
    k::Int = 10,
    target_recall::Float64 = 0.95,
    selection_margin::Float64 = 0.02,
    probe_safety_factor::Float64 = 2.0,
    repetitions::Int = 3,
    iterations::Int = 8,
    restarts::Int = 2,
    training_count::Int = min(size(dataset.vectors, 2), 20_000),
    candidate_multiplier::Float64 = 16.0,
    postfilter_safety_factor::Float64 = 2.0,
    vector_weight::Float64 = 0.5,
    filter_weight::Float64 = 0.5,
    rerank_factor::Int = 4,
    seed::Int = dataset.manifest.sampling_seed,
)
    selectivities=dataset_selectivities(
        dataset.metadata,
        vcat(dataset.train_filters, dataset.heldout_filters),
    )
    average_selectivity=sum(selectivities)/length(selectivities)
    return ExperimentSpec(
        name;
        vector_count = size(dataset.vectors, 2),
        dimension = size(dataset.vectors, 1),
        train_query_count = size(dataset.train_queries, 2),
        heldout_query_count = size(dataset.heldout_queries, 2),
        nlists = nlists,
        nprobes = nprobes,
        k = k,
        target_recall = target_recall,
        selection_margin = selection_margin,
        probe_safety_factor = probe_safety_factor,
        candidate_multiplier = candidate_multiplier,
        postfilter_safety_factor = postfilter_safety_factor,
        vector_weight = vector_weight,
        filter_weight = filter_weight,
        rerank_factor = rerank_factor,
        repetitions = repetitions,
        iterations = iterations,
        restarts = restarts,
        training_count = training_count,
        metric = :cosine,
        vector_workload = :real,
        filter_workload = :natural,
        selectivity = average_selectivity,
        seed = seed,
    )
end

function run_real_benchmark(
    dataset::ArxivFANNSDataset;
    name::AbstractString = "arxiv-fanns-100k",
    nlists::Int = 64,
    nprobes::AbstractVector{<:Integer} = [1, 2, 4, 8, 16, 32, 64],
    k::Int = 10,
    target_recall::Float64 = 0.95,
    selection_margin::Float64 = 0.02,
    probe_safety_factor::Float64 = 2.0,
    repetitions::Int = 3,
    iterations::Int = 8,
    restarts::Int = 2,
    training_count::Int = min(size(dataset.vectors, 2), 20_000),
    candidate_multiplier::Float64 = 16.0,
    postfilter_safety_factor::Float64 = 2.0,
    vector_weight::Float64 = 0.5,
    filter_weight::Float64 = 0.5,
    rerank_factor::Int = 4,
    seed::Int = dataset.manifest.sampling_seed,
)
    spec=real_experiment_spec(
        dataset;
        name = name,
        nlists = nlists,
        nprobes = nprobes,
        k = k,
        target_recall = target_recall,
        selection_margin = selection_margin,
        probe_safety_factor = probe_safety_factor,
        repetitions = repetitions,
        iterations = iterations,
        restarts = restarts,
        training_count = training_count,
        candidate_multiplier = candidate_multiplier,
        postfilter_safety_factor = postfilter_safety_factor,
        vector_weight = vector_weight,
        filter_weight = filter_weight,
        rerank_factor = rerank_factor,
        seed = seed,
    )
    context=nothing
    build_seconds=@elapsed context=build_benchmark_context(
        dataset.vectors,
        dataset.metadata;
        nlists = spec.nlists,
        iterations = spec.iterations,
        seed = spec.seed,
        metric = spec.metric,
        restarts = spec.restarts,
        training_count = spec.training_count,
    )
    train_evaluations=Dict{Symbol,Dict{Int,Any}}()

    for method in (:ivf_prefilter, :filter_aware, :filter_aware_bound)
        method_evaluations=Dict{Int,Any}()
        for nprobe in spec.nprobes
            method_evaluations[nprobe]=evaluate_benchmark_method(
                spec,
                context,
                dataset.vectors,
                dataset.metadata,
                dataset.train_queries,
                dataset.train_filters,
                dataset.train_truth,
                method,
                nprobe;
                split = :train,
            )
        end
        train_evaluations[method]=method_evaluations
    end

    selections=[
        select_real_method(spec, train_evaluations[method], method) for
        method in (:ivf_prefilter, :filter_aware, :filter_aware_bound)
    ]
    ranking_seconds=@elapsed rankings=build_postfilter_rankings(
        spec,
        context.index.ivf,
        dataset.vectors,
        dataset.train_queries,
    )
    postfilter_selection=select_postfilter_configuration(
        spec,
        context,
        dataset.vectors,
        dataset.metadata,
        dataset.train_queries,
        dataset.train_filters,
        dataset.train_truth;
        prebuilt_rankings = rankings,
    )
    push!(selections, postfilter_selection)
    push!(
        selections,
        (
            method = :exact,
            nprobe = 0,
            train_recall = 1.0,
            train_p50_ms = NaN,
            train_p95_ms = NaN,
            train_candidates = sum(
                count_exact_candidates(context.filter_index, filter) for
                filter in dataset.train_filters
            )/length(dataset.train_filters),
            achieved = true,
            selection_target = spec.target_recall,
            postfilter_candidate_multiplier = 0.0,
        ),
    )
    sort!(
        selections;
        by = selection->findfirst(==(selection.method), experiment_methods(spec)),
    )
    calibration=RecallCalibration()

    for selection in selections
        selection.method===:exact&&continue
        add_calibration_entry!(
            calibration,
            RecallCalibrationEntry(
                selection.method,
                spec.filter_workload,
                spec.selectivity,
                spec.vector_count,
                spec.nlists,
                spec.target_recall,
                selection.nprobe,
                selection.train_recall,
                selection.train_p50_ms,
                selection.achieved,
                spec.dimension,
                spec.metric,
                1,
            ),
        )
    end

    planner_config=PlannerConfig(
        target_recall = spec.target_recall,
        candidate_multiplier = spec.candidate_multiplier,
        postfilter_candidate_multiplier = postfilter_selection.postfilter_candidate_multiplier,
        default_nprobe = min(4, spec.nlists),
        max_nprobe = maximum(spec.nprobes),
        calibration = calibration,
        workload = spec.filter_workload,
    )
    db=benchmark_database(context, dataset.vectors, dataset.metadata, spec.metric)
    method_results=Dict{Symbol,Any}()
    raw_rows=NamedTuple[]

    for selection in selections
        result=evaluate_benchmark_method(
            spec,
            context,
            dataset.vectors,
            dataset.metadata,
            dataset.heldout_queries,
            dataset.heldout_filters,
            dataset.heldout_truth,
            selection.method,
            selection.nprobe;
            split = :heldout,
            postfilter_candidate_multiplier = selection.postfilter_candidate_multiplier,
        )
        method_results[selection.method]=result
    end

    auto_result=evaluate_benchmark_method(
        spec,
        context,
        dataset.vectors,
        dataset.metadata,
        dataset.heldout_queries,
        dataset.heldout_filters,
        dataset.heldout_truth,
        :auto,
        0;
        split = :heldout,
        db = db,
        planner_config = planner_config,
    )
    method_results[:auto]=auto_result
    selectivities=dataset_selectivities(dataset.metadata, dataset.heldout_filters)

    for method in vcat(experiment_methods(spec), [:auto])
        for row in method_results[method].rows
            query_index=row.query_index
            push!(
                raw_rows,
                merge(
                    (
                        dataset = dataset.manifest.name,
                        source_query_index = dataset.heldout_indices[query_index],
                        query_selectivity = selectivities[query_index],
                    ),
                    row,
                ),
            )
        end
    end

    eligible=[
        (method = method, evaluation = method_results[method]) for
        method in experiment_methods(spec) if
        method_results[method].summary.average_recall>=spec.target_recall
    ]
    oracle=isempty(eligible) ? nothing :
           first(sort(eligible; by = value->value.evaluation.summary.latency.p95_ms))
    planner_regret=oracle===nothing ? Inf :
                   auto_result.summary.latency.p95_ms/oracle.evaluation.summary.latency.p95_ms
    approximate_methods=filter(!=(:exact), experiment_methods(spec))
    best_approximate_recall=maximum(
        method_results[method].summary.average_recall for method in approximate_methods
    )
    recall_gate=best_approximate_recall>=spec.target_recall
    planner_gate=auto_result.summary.average_recall>=spec.target_recall
    source_bytes=sum(values(dataset.manifest.file_sizes))

    return (
        spec = spec,
        dataset_manifest = dataset.manifest,
        build_seconds = build_seconds,
        index_megabytes = Base.summarysize(context)/1024^2,
        source_megabytes = source_bytes/1024^2,
        maxrss_megabytes = Sys.maxrss()/1024^2,
        postfilter_ranking_seconds = ranking_seconds,
        postfilter_ranking_megabytes = Base.summarysize(rankings)/1024^2,
        selections = selections,
        method_results = method_results,
        raw_rows = raw_rows,
        train_query_indices = copy(dataset.train_indices),
        heldout_query_indices = copy(dataset.heldout_indices),
        oracle_method = oracle===nothing ? :none : oracle.method,
        planner_regret = planner_regret,
        best_approximate_recall = best_approximate_recall,
        recall_gate = recall_gate,
        planner_gate = planner_gate,
        passed = recall_gate&&planner_gate,
    )
end

function real_benchmark_summary_rows(result)
    selection_map=Dict(selection.method=>selection for selection in result.selections)
    rows=NamedTuple[]

    for method in vcat(experiment_methods(result.spec), [:auto])
        evaluation=result.method_results[method]
        selection=get(selection_map, method, nothing)
        push!(
            rows,
            (
                dataset = result.dataset_manifest.name,
                experiment = result.spec.name,
                vector_count = result.spec.vector_count,
                dimension = result.spec.dimension,
                train_queries = result.spec.train_query_count,
                heldout_queries = result.spec.heldout_query_count,
                method = String(method),
                selected_nprobe = selection===nothing ? 0 : selection.nprobe,
                train_recall = selection===nothing ? NaN : selection.train_recall,
                heldout_recall = evaluation.summary.average_recall,
                p50_ms = evaluation.summary.latency.p50_ms,
                p95_ms = evaluation.summary.latency.p95_ms,
                qps = benchmark_qps(evaluation),
                candidates_visited = evaluation.summary.average_candidates_visited,
                candidates_scored = evaluation.summary.average_candidates_scored,
                lists_probed = evaluation.summary.average_lists_probed,
                build_seconds = result.build_seconds,
                index_megabytes = result.index_megabytes,
                source_megabytes = result.source_megabytes,
                peak_rss_megabytes = result.maxrss_megabytes,
                target_recall = result.spec.target_recall,
                recall_gate = result.recall_gate,
                planner_gate = result.planner_gate,
                oracle_method = String(result.oracle_method),
                planner_regret = result.planner_regret,
            ),
        )
    end

    return rows
end

function save_real_benchmark(path::AbstractString, result)
    mkpath(path)
    raw=write_tsv(joinpath(path, "raw_results.tsv"), result.raw_rows)
    summary=write_tsv(joinpath(path, "summary.tsv"), real_benchmark_summary_rows(result))
    dataset_manifest=save_dataset_manifest(
        joinpath(path, "dataset.toml"),
        result.dataset_manifest,
    )
    environment_path=joinpath(path, "environment.toml")
    environment=Dict(
        "julia_version"=>string(VERSION),
        "julia_threads"=>Threads.nthreads(),
        "kernel"=>string(Sys.KERNEL),
        "architecture"=>string(Sys.ARCH),
        "cpu_threads"=>Sys.CPU_THREADS,
        "experiment"=>result.spec.name,
        "dataset_hash"=>result.dataset_manifest.preprocessing_hash,
        "target_recall"=>result.spec.target_recall,
        "nlists"=>result.spec.nlists,
        "nprobes"=>result.spec.nprobes,
        "k"=>result.spec.k,
        "train_query_indices"=>result.train_query_indices,
        "heldout_query_indices"=>result.heldout_query_indices,
        "recall_gate"=>result.recall_gate,
        "planner_gate"=>result.planner_gate,
        "passed"=>result.passed,
    )
    open(environment_path, "w") do io
        TOML.print(io, environment)
    end
    return (
        raw = raw,
        summary = summary,
        dataset = dataset_manifest,
        environment = environment_path,
    )
end
