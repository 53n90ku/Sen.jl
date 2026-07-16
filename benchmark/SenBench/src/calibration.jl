function add_sweep_calibration!(
    calibration::RecallCalibration,
    sweep_results::AbstractVector;
    workload::Symbol,
    selectivity::Float64,
    vector_count::Int,
    list_count::Int,
    dimension::Int = 0,
    metric::Symbol = :cosine,
    index_version::Int = 1,
    selection_margin::Float64 = 0.0,
    probe_safety_factor::Float64 = 1.0,
    target_recalls::AbstractVector{<:Real} = [0.80, 0.90, 0.95],
    methods::AbstractVector{Symbol} = [:ivf_prefilter, :ivf_postfilter, :filter_aware],
)
    isempty(sweep_results)&&throw(ArgumentError("sweep results cannot be empty"))
    workload in (:random, :correlated, :anticorrelated, :skewed)||throw(
        ArgumentError("unsupported calibration workload"),
    )
    0.0<=selectivity<=1.0||throw(ArgumentError("selectivity must be between 0 and 1"))
    vector_count>0||throw(ArgumentError("vector count must be positive"))
    list_count>0||throw(ArgumentError("list count must be positive"))
    dimension>=0||throw(ArgumentError("dimension cannot be negative"))
    metric in (:cosine, :dot)||throw(ArgumentError("unsupported calibration metric"))
    index_version>0||throw(ArgumentError("index version must be positive"))
    selection_margin>=0||throw(ArgumentError("selection margin cannot be negative"))
    probe_safety_factor>=1.0||throw(ArgumentError("probe safety factor must be atleast 1"))

    for method in methods
        points=sweep_points(sweep_results, method)
        isempty(points)&&continue

        for target in target_recalls
            target_recall=Float64(target)
            0.0<target_recall<=1.0||throw(
                ArgumentError("target recall must be between 0 and 1"),
            )
            selection_target=min(1.0, target_recall+selection_margin)
            selected=conservative_at_recall(
                sweep_results,
                method,
                selection_target;
                probe_safety_factor = probe_safety_factor,
            )

            if selected===nothing&&selection_target>target_recall
                selected=conservative_at_recall(
                    sweep_results,
                    method,
                    target_recall;
                    probe_safety_factor = probe_safety_factor,
                )
            end

            achieved=selected!==nothing&&selected.recall>=target_recall

            if selected===nothing
                ranked=sort(points; by = point->(-point.recall, point.p50_ms, point.nprobe))
                selected=first(ranked)
            end

            entry=RecallCalibrationEntry(
                method,
                workload,
                selectivity,
                vector_count,
                list_count,
                target_recall,
                selected.nprobe,
                selected.recall,
                selected.p50_ms,
                achieved,
                dimension,
                metric,
                index_version,
            )
            add_calibration_entry!(calibration, entry)
        end
    end

    return calibration
end

function calibrate_recall(
    sweep_results::AbstractVector;
    workload::Symbol,
    selectivity::Float64,
    vector_count::Int,
    list_count::Int,
    dimension::Int = 0,
    metric::Symbol = :cosine,
    index_version::Int = 1,
    selection_margin::Float64 = 0.0,
    probe_safety_factor::Float64 = 1.0,
    target_recalls::AbstractVector{<:Real} = [0.80, 0.90, 0.95],
    methods::AbstractVector{Symbol} = [:ivf_prefilter, :ivf_postfilter, :filter_aware],
)
    calibration=RecallCalibration()
    return add_sweep_calibration!(
        calibration,
        sweep_results;
        workload = workload,
        selectivity = selectivity,
        vector_count = vector_count,
        list_count = list_count,
        dimension = dimension,
        metric = metric,
        index_version = index_version,
        selection_margin = selection_margin,
        probe_safety_factor = probe_safety_factor,
        target_recalls = target_recalls,
        methods = methods,
    )
end

function evaluate_recall_calibration(
    calibration::RecallCalibration,
    context,
    vectors::AbstractMatrix,
    metadata::AbstractVector,
    queries::AbstractMatrix,
    filters::AbstractVector,
    method::Symbol;
    workload::Symbol,
    selectivity::Float64,
    target_recall::Float64 = 0.90,
    k::Int = 10,
    metric::Symbol = :cosine,
    repetitions::Int = 1,
    postfilter_oversample::Int = 10,
    candidate_multiplier::Float64 = 4.0,
    vector_weight::Float64 = 0.5,
    filter_weight::Float64 = 0.5,
    rerank_factor::Int = 4,
)
    _, query_count=size(queries)
    length(filters)==query_count||throw(
        DimensionMismatch("filter count doesnt match queries"),
    )
    query_count>0||throw(ArgumentError("queries cannot be empty"))

    lookup=lookup_calibrated_nprobe(
        calibration,
        method;
        workload = workload,
        selectivity = selectivity,
        vector_count = size(vectors, 2),
        list_count = length(context.index.ivf.lists),
        dimension = size(vectors, 1),
        metric = metric,
        target_recall = target_recall,
    )
    lookup===nothing&&throw(ArgumentError("calibration method is missing"))

    recalls=Float64[]
    times_ns=Int[]

    for query_index = 1:query_count
        query=@view queries[:, query_index]
        filter=filters[query_index]
        truth=search_exact(
            vectors,
            metadata,
            query;
            k = k,
            metric = metric,
            filter = filter,
            filter_index = context.filter_index,
            vector_norms = context.index.ivf.vector_norms,
        )
        truth_ids=Int[result.index for result in truth]
        resolved_oversample=resolve_postfilter_oversample(
            postfilter_oversample,
            selectivity;
            candidate_multiplier = candidate_multiplier,
        )

        search_function=if method===:ivf_prefilter
            ()->search_ivf_prefilter(
                context.index,
                vectors,
                metadata,
                query;
                k = k,
                nprobe = lookup.nprobe,
                metric = metric,
                filter = filter,
            )
        elseif method===:ivf_postfilter
            ()->search_ivf_postfilter(
                context.index.ivf,
                vectors,
                metadata,
                query;
                k = k,
                nprobe = lookup.nprobe,
                metric = metric,
                filter = filter,
                oversample = resolved_oversample,
            )
        elseif method===:filter_aware
            ()->search_filter_aware_ivf(
                context.index,
                vectors,
                metadata,
                query;
                k = k,
                nprobe = lookup.nprobe,
                metric = metric,
                filter = filter,
                adaptive = false,
                vector_weight = vector_weight,
                filter_weight = filter_weight,
                rerank_factor = rerank_factor,
            )
        elseif method===:filter_aware_bound
            ()->search_filter_aware_bound(
                context.index,
                vectors,
                metadata,
                query;
                k = k,
                minimum_nprobe = 1,
                max_nprobe = lookup.nprobe,
                metric = metric,
                filter = filter,
            )
        else
            throw(ArgumentError("unsupported calibration method"))
        end

        measurement=measure_latency(search_function; repetitions = repetitions)
        predicted_ids=Int[result.index for result in measurement.result]
        push!(recalls, recall_at_k(predicted_ids, truth_ids, k))
        append!(times_ns, measurement.times_ns)
    end

    average_recall=sum(recalls)/length(recalls)

    return (
        method = method,
        workload = workload,
        selectivity = selectivity,
        target_recall = target_recall,
        nprobe = lookup.nprobe,
        average_recall = average_recall,
        passed = average_recall>=target_recall,
        latency = latency_summary(times_ns),
    )
end
