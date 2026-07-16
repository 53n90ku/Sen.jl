"""
    PlannerConfig(; kwargs...)

Tune automatic query planning, including recall targets, candidate expansion,
probe limits and the cost model used by [`plan_query`](@ref).
"""
struct PlannerConfig
    target_recall::Float64
    candidate_multiplier::Float64
    postfilter_candidate_multiplier::Float64
    default_nprobe::Int
    max_nprobe::Int
    centroid_cost::Float64
    metadata_check_cost::Float64
    vector_score_cost::Float64
    filter_aware_gain::Float64
    exact_filter_threshold::Float64
    calibration::Union{Nothing,RecallCalibration}
    workload::Symbol
end

function PlannerConfig(;
    target_recall::Float64 = 0.90,
    candidate_multiplier::Float64 = 4.0,
    postfilter_candidate_multiplier::Float64 = candidate_multiplier,
    default_nprobe::Int = 4,
    max_nprobe::Int = 64,
    centroid_cost::Float64 = 0.05,
    metadata_check_cost::Float64 = 0.10,
    vector_score_cost::Float64 = 1.0,
    filter_aware_gain::Float64 = 0.75,
    exact_filter_threshold::Float64 = 0.01,
    calibration::Union{Nothing,RecallCalibration} = nothing,
    workload::Symbol = :auto,
)
    0.0<target_recall<=1.0||throw(ArgumentError("target recall must be between 0 and 1"))
    candidate_multiplier>0||throw(ArgumentError("candidate multiplier must be positive"))
    postfilter_candidate_multiplier>0||throw(
        ArgumentError("postfilter candidate multiplier must be positive"),
    )
    default_nprobe>0||throw(ArgumentError("default nprobe must be positive"))
    max_nprobe>=default_nprobe||throw(
        ArgumentError("max nprobe cannot be smaller than default nprobe"),
    )
    centroid_cost>=0||throw(ArgumentError("centroid cost cannot be negative"))
    metadata_check_cost>=0||throw(ArgumentError("metadata check cost cannot be negative"))
    vector_score_cost>=0||throw(ArgumentError("vector score cost cannot be negative"))
    0.0<=filter_aware_gain<=1.0||throw(
        ArgumentError("filter aware gain must be between 0 and 1"),
    )
    0.0<=exact_filter_threshold<=1.0||throw(
        ArgumentError("exact filter threshold must be between 0 and 1"),
    )
    workload in (:auto, :random, :correlated, :anticorrelated, :skewed, :natural)||throw(
        ArgumentError("invalid workload"),
    )

    return PlannerConfig(
        target_recall,
        candidate_multiplier,
        postfilter_candidate_multiplier,
        default_nprobe,
        max_nprobe,
        centroid_cost,
        metadata_check_cost,
        vector_score_cost,
        filter_aware_gain,
        exact_filter_threshold,
        calibration,
        workload,
    )
end

function estimate_candidate_count(total_count::Int, selectivity::Float64)
    total_count>=0||throw(ArgumentError("total count cannot be negative"))
    0.0<=selectivity<=1.0||throw(ArgumentError("selectivity must be between 0 and 1"))
    return ceil(Int, total_count*selectivity)
end

function estimate_required_nprobe(
    total_count::Int,
    nlists::Int,
    selectivity::Float64,
    k::Int;
    candidate_multiplier::Float64 = 4.0,
    target_recall::Float64 = 0.90,
    max_nprobe::Int = nlists,
)
    total_count>0||throw(ArgumentError("total count must be positive"))
    nlists>0||throw(ArgumentError("nlists must be positive"))
    0.0<=selectivity<=1.0||throw(ArgumentError("selectivity must be between 0 and 1"))
    k>0||throw(ArgumentError("k must be positive"))
    candidate_multiplier>0||throw(ArgumentError("candidate multiplier must be positive"))
    0.0<target_recall<=1.0||throw(ArgumentError("target recall must be between 0 and 1"))
    1<=max_nprobe<=nlists||throw(ArgumentError("max nprobe must be between 1 and nlists"))

    selectivity==0&&return max_nprobe

    target_candidates=ceil(Int, k*candidate_multiplier/target_recall)
    matches_per_list=total_count*selectivity/nlists
    required=ceil(Int, target_candidates/matches_per_list)

    return clamp(required, 1, max_nprobe)
end

function logical_filter_stats(db::VectorDB, filter::Union{NamedTuple,FilterExpr})
    expression=normalize_filter(filter)
    base_metadata=stored_metadata(db.metadata_store)
    base_mask=if db.filter_index!==nothing&&supports_indexed_filter(
        db.filter_index,
        expression,
    )
        evaluate_filter(db.filter_index, expression)
    else
        BitVector(matches_filter(row, expression) for row in base_metadata)
    end

    length(base_mask)==length(db.base_tombstones)||throw(
        DimensionMismatch("base filter mask doesnt match tombstones"),
    )
    base_live_count=length(base_mask)-count(db.base_tombstones)
    base_matching_count=count(
        index->base_mask[index]&&!db.base_tombstones[index],
        eachindex(base_mask),
    )
    delta_metadata=stored_metadata(db.delta_store.metadata_store)
    delta_matching_count=count(row->matches_filter(row, expression), delta_metadata)
    delta_count=length(delta_metadata)
    logical_count=base_live_count+delta_count
    matching_count=base_matching_count+delta_matching_count
    logical_selectivity=logical_count==0 ? 0.0 : matching_count/logical_count
    base_selectivity=base_live_count==0 ? 0.0 : base_matching_count/base_live_count

    return (
        base_live_count = base_live_count,
        base_matching_count = base_matching_count,
        base_selectivity = base_selectivity,
        delta_count = delta_count,
        delta_matching_count = delta_matching_count,
        logical_count = logical_count,
        matching_count = matching_count,
        logical_selectivity = logical_selectivity,
    )
end

function estimate_filter_concentration(
    index::FilterAwareIVFIndex,
    filter::Union{NamedTuple,FilterExpr};
    metadata::Union{Nothing,AbstractVector} = nothing,
    excluded::Union{Nothing,BitVector} = nothing,
)
    filter=normalize_filter(filter)
    excluded===nothing||length(excluded)==sum(length, index.ivf.lists)||throw(
        DimensionMismatch("excluded mask doesnt match index"),
    )
    metadata===nothing||length(metadata)==sum(length, index.ivf.lists)||throw(
        DimensionMismatch("metadata count doesnt match index"),
    )
    matching_counts=Int[]

    for list_index in eachindex(index.ivf.lists)
        list=index.ivf.lists[list_index]
        local_index=index.metadata_indexes[list_index]
        mask=if supports_indexed_filter(local_index, filter)
            evaluate_filter(local_index, filter)
        else
            metadata===nothing&&throw(
                ArgumentError(
                    "filter concentration requires metadata because this expression is not supported by the list metadata index",
                ),
            )
            BitVector(
                matches_filter(metadata[vector_index], filter) for vector_index in list
            )
        end

        push!(
            matching_counts,
            count(
                position->mask[position]&&(excluded===nothing||!excluded[list[position]]),
                eachindex(list),
            ),
        )
    end

    total_matches=sum(matching_counts)
    total_matches==0&&return 0.0

    nonempty_lists=count(list->!isempty(list), index.ivf.lists)
    nonempty_lists<=1&&return 1.0

    shares=matching_counts ./ total_matches
    hhi=sum(abs2, shares)
    uniform_hhi=1/nonempty_lists

    return clamp((hhi-uniform_hhi)/(1-uniform_hhi), 0.0, 1.0)
end

function estimate_strategy_costs(
    db::VectorDB,
    filter::Union{NamedTuple,FilterExpr};
    k::Int = 10,
    config::PlannerConfig = PlannerConfig(),
)
    filter=normalize_filter(filter)
    filter_index=db.filter_index
    filter_index===nothing&&throw(
        ArgumentError("filter index must be built before planning"),
    )

    stats=logical_filter_stats(db, filter)
    stats.logical_count>0||throw(ArgumentError("database cannot be empty"))
    stats.base_live_count>0||throw(ArgumentError("database has no indexed live vectors"))

    total_count=stats.base_live_count
    selectivity=stats.logical_selectivity
    base_selectivity=stats.base_selectivity
    matching_count=stats.matching_count
    index=db.index
    estimated_nlists=index===nothing ?
                     min(config.max_nprobe, max(1, ceil(Int, sqrt(total_count)))) :
                     length(index.ivf.lists)
    probe_limit=min(config.max_nprobe, estimated_nlists)
    uniform_nprobe=estimate_required_nprobe(
        total_count,
        estimated_nlists,
        base_selectivity,
        k;
        candidate_multiplier = config.candidate_multiplier,
        target_recall = config.target_recall,
        max_nprobe = probe_limit,
    )

    concentration=index===nothing ? 0.0 :
                  estimate_filter_concentration(
        index,
        filter;
        metadata = stored_metadata(db.metadata_store),
        excluded = db.base_tombstones,
    )
    workload=config.workload===:auto ? classify_filter_workload(concentration) :
             config.workload
    concentration_penalty=max(0.10, 1-concentration)
    required_nprobe=min(probe_limit, ceil(Int, uniform_nprobe/concentration_penalty))
    aware_nprobe=max(
        1,
        ceil(Int, uniform_nprobe*(1-config.filter_aware_gain*concentration)),
    )

    prefilter_calibration=config.calibration===nothing ? nothing :
                          lookup_calibrated_nprobe(
        config.calibration,
        :ivf_prefilter;
        workload = workload,
        selectivity = selectivity,
        vector_count = total_count,
        list_count = estimated_nlists,
        dimension = db.dim,
        metric = db.metric,
        target_recall = config.target_recall,
    )
    postfilter_calibration=config.calibration===nothing ? nothing :
                           lookup_calibrated_nprobe(
        config.calibration,
        :ivf_postfilter;
        workload = workload,
        selectivity = selectivity,
        vector_count = total_count,
        list_count = estimated_nlists,
        dimension = db.dim,
        metric = db.metric,
        target_recall = config.target_recall,
    )
    aware_calibration=config.calibration===nothing ? nothing :
                      lookup_calibrated_nprobe(
        config.calibration,
        :filter_aware;
        workload = workload,
        selectivity = selectivity,
        vector_count = total_count,
        list_count = estimated_nlists,
        dimension = db.dim,
        metric = db.metric,
        target_recall = config.target_recall,
    )
    bound_calibration=config.calibration===nothing ? nothing :
                      lookup_calibrated_nprobe(
        config.calibration,
        :filter_aware_bound;
        workload = workload,
        selectivity = selectivity,
        vector_count = total_count,
        list_count = estimated_nlists,
        dimension = db.dim,
        metric = db.metric,
        target_recall = config.target_recall,
    )

    prefilter_nprobe=prefilter_calibration===nothing ? required_nprobe :
                     prefilter_calibration.nprobe
    postfilter_nprobe=postfilter_calibration===nothing ? required_nprobe :
                      postfilter_calibration.nprobe
    filter_aware_min_nprobe=aware_calibration===nothing ?
                            min(config.default_nprobe, aware_nprobe) :
                            aware_calibration.nprobe
    filter_aware_max_nprobe=aware_calibration===nothing ?
                            max(aware_nprobe, required_nprobe) : aware_calibration.nprobe
    filter_aware_bound_min_nprobe=1
    filter_aware_bound_max_nprobe=bound_calibration===nothing ? required_nprobe :
                                  bound_calibration.nprobe
    average_list_size=total_count/estimated_nlists
    centroid_work=estimated_nlists*db.dim*config.centroid_cost
    bitset_work=ceil(Int, total_count/64)*config.metadata_check_cost

    prefilter_visited=min(total_count, ceil(Int, prefilter_nprobe*average_list_size))
    prefilter_scored=min(
        stats.base_matching_count,
        ceil(Int, prefilter_visited*base_selectivity),
    )
    postfilter_scored=min(total_count, ceil(Int, postfilter_nprobe*average_list_size))

    aware_visited=min(total_count, ceil(Int, filter_aware_min_nprobe*average_list_size))
    concentration_boost=1+concentration*(estimated_nlists-1)
    aware_density=min(1.0, base_selectivity*concentration_boost)
    aware_scored=min(stats.base_matching_count, ceil(Int, aware_visited*aware_density))
    bound_visited=min(
        stats.base_matching_count,
        ceil(Int, filter_aware_bound_max_nprobe*average_list_size*aware_density),
    )
    bound_scored=bound_visited

    delta_cost=stats.delta_count*config.metadata_check_cost+stats.delta_matching_count*db.dim*config.vector_score_cost
    exact_cost=bitset_work+matching_count*db.dim*config.vector_score_cost+stats.delta_count*config.metadata_check_cost
    prefilter_cost=centroid_work+prefilter_visited*config.metadata_check_cost+prefilter_scored*db.dim*config.vector_score_cost+delta_cost
    postfilter_cost=centroid_work+postfilter_scored*db.dim*config.vector_score_cost+delta_cost
    aware_cost=centroid_work+estimated_nlists*config.metadata_check_cost+aware_visited*config.metadata_check_cost+aware_scored*db.dim*config.vector_score_cost+delta_cost
    bound_cost=centroid_work+estimated_nlists*config.metadata_check_cost+bound_scored*db.dim*config.vector_score_cost+delta_cost

    return (
        selectivity = selectivity,
        base_selectivity = base_selectivity,
        matching_count = matching_count,
        base_matching_count = stats.base_matching_count,
        delta_matching_count = stats.delta_matching_count,
        concentration = concentration,
        workload = workload,
        uniform_nprobe = uniform_nprobe,
        required_nprobe = required_nprobe,
        aware_nprobe = aware_nprobe,
        prefilter_nprobe = prefilter_nprobe,
        postfilter_nprobe = postfilter_nprobe,
        filter_aware_min_nprobe = filter_aware_min_nprobe,
        filter_aware_max_nprobe = filter_aware_max_nprobe,
        filter_aware_bound_min_nprobe = filter_aware_bound_min_nprobe,
        filter_aware_bound_max_nprobe = filter_aware_bound_max_nprobe,
        prefilter_calibrated = prefilter_calibration!==nothing,
        postfilter_calibrated = postfilter_calibration!==nothing,
        filter_aware_calibrated = aware_calibration!==nothing,
        filter_aware_bound_calibrated = bound_calibration!==nothing,
        exact = exact_cost,
        ivf_prefilter = prefilter_cost,
        ivf_postfilter = postfilter_cost,
        filter_aware = aware_cost,
        filter_aware_bound = bound_cost,
    )
end
