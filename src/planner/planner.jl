struct QueryPlan
    strategy::SearchStrategy
    selectivity::Float64
    estimated_candidates::Int
    minimum_nprobe::Int
    nprobe::Int
    estimated_cost::Float64
    costs::NamedTuple
    reason::String
end

const PLAN_CACHE_LIMIT=1024

function planner_config_key(config::PlannerConfig)
    calibration_key=config.calibration===nothing ? UInt(0) :
                    hash(config.calibration.entries)
    return (
        config.target_recall,
        config.candidate_multiplier,
        config.postfilter_candidate_multiplier,
        config.default_nprobe,
        config.max_nprobe,
        config.centroid_cost,
        config.metadata_check_cost,
        config.vector_score_cost,
        config.filter_aware_gain,
        config.exact_filter_threshold,
        calibration_key,
        config.workload,
    )
end

function cached_plan_query(
    db::VectorDB,
    filter::Union{Nothing,NamedTuple,FilterExpr};
    k::Int = 10,
    config::PlannerConfig = PlannerConfig(),
)
    filter=normalize_filter(filter)

    return with_database_read(db.database_lock) do
        key=(db.revision, db.index_revision, filter, k, planner_config_key(config))
        lock(db.plan_cache_lock)

        try
            haskey(db.plan_cache, key)&&return db.plan_cache[key]::QueryPlan
            length(db.plan_cache)>=PLAN_CACHE_LIMIT&&empty!(db.plan_cache)
            plan=plan_query(db, filter; k = k, config = config)
            db.plan_cache[key]=plan
            return plan
        finally
            unlock(db.plan_cache_lock)
        end
    end
end

function strategy_name(strategy::SearchStrategy)
    strategy isa PreFilterExactStrategy&&return :exact
    strategy isa IVFPreFilterStrategy&&return :ivf_prefilter
    strategy isa IVFPostFilterStrategy&&return :ivf_postfilter
    strategy isa FilterAwareIVFStrategy&&return :filter_aware
    strategy isa BoundedFilterAwareIVFStrategy&&return :filter_aware_bound
    strategy isa ExactStrategy&&return :exact
    return :ivf
end

function exact_query_plan(
    db::VectorDB,
    filter::Union{Nothing,FilterExpr};
    reason::String = "exact search was selected",
    config::PlannerConfig = PlannerConfig(),
)
    if filter===nothing
        matching_count=length(db)
        selectivity=matching_count==0 ? 0.0 : 1.0
        estimated_cost=matching_count*db.dim*config.vector_score_cost
    else
        stats=logical_filter_stats(db, filter)
        matching_count=stats.matching_count
        selectivity=stats.logical_selectivity
        estimated_cost=stats.logical_count*config.metadata_check_cost+matching_count*db.dim*config.vector_score_cost
    end

    costs=(exact = estimated_cost,)
    strategy=filter===nothing ? ExactStrategy() : PreFilterExactStrategy()
    return QueryPlan(
        strategy,
        selectivity,
        matching_count,
        0,
        0,
        estimated_cost,
        costs,
        reason,
    )
end

function supports_list_filter(index::FilterAwareIVFIndex, filter::FilterExpr)
    return all(
        metadata_index->supports_indexed_filter(metadata_index, filter),
        index.metadata_indexes,
    )
end

function choose_strategy(
    db::VectorDB,
    filter::Union{Nothing,NamedTuple,FilterExpr};
    k::Int = 10,
    config::PlannerConfig = PlannerConfig(),
)
    filter=normalize_filter(filter)
    k>0||throw(ArgumentError("k must be positive"))

    if filter===nothing
        db.index===nothing&&return exact_query_plan(
            db,
            nothing;
            reason = "database has no built index",
            config = config,
        )
        list_count=db.index===nothing ? config.max_nprobe : length(db.index.ivf.lists)
        nprobe=min(config.default_nprobe, list_count)
        costs=(ivf = Float64(length(db)*db.dim),)
        return QueryPlan(
            IVFStrategy(),
            1.0,
            length(db),
            nprobe,
            nprobe,
            costs.ivf,
            costs,
            "query has no filter",
        )
    end

    db.index===nothing&&return exact_query_plan(
        db,
        filter;
        reason = "database has no built index",
        config = config,
    )
    stats=logical_filter_stats(db, filter)
    stats.base_live_count==0&&return exact_query_plan(
        db,
        filter;
        reason = "no indexed live vectors remain",
        config = config,
    )
    supports_list_filter(db.index, filter)||return exact_query_plan(
        db,
        filter;
        reason = "filter expression requires exact scalar evaluation",
        config = config,
    )

    estimates=estimate_strategy_costs(db, filter; k = k, config = config)

    if estimates.matching_count==0
        return QueryPlan(
            PreFilterExactStrategy(),
            estimates.selectivity,
            0,
            0,
            0,
            estimates.exact,
            estimates,
            "filter has no matches",
        )
    end

    if estimates.selectivity<=config.exact_filter_threshold
        return QueryPlan(
            PreFilterExactStrategy(),
            estimates.selectivity,
            estimates.matching_count,
            0,
            0,
            estimates.exact,
            estimates,
            "rare filter is cheaper and safer with exact search",
        )
    end

    choices=[
        (
            strategy = PreFilterExactStrategy(),
            cost = estimates.exact,
            minimum_nprobe = 0,
            nprobe = 0,
            reason = "exact filtering has the lowest estimated work",
        ),
        (
            strategy = IVFPreFilterStrategy(),
            cost = estimates.ivf_prefilter,
            minimum_nprobe = estimates.prefilter_nprobe,
            nprobe = estimates.prefilter_nprobe,
            reason = "vector-first lists with prefiltering have the lowest estimated work",
        ),
        (
            strategy = IVFPostFilterStrategy(),
            cost = estimates.ivf_postfilter,
            minimum_nprobe = estimates.postfilter_nprobe,
            nprobe = estimates.postfilter_nprobe,
            reason = "postfiltering has the lowest estimated work",
        ),
        (
            strategy = FilterAwareIVFStrategy(),
            cost = estimates.filter_aware,
            minimum_nprobe = estimates.filter_aware_min_nprobe,
            nprobe = estimates.filter_aware_max_nprobe,
            reason = "filter concentration makes metadata-aware probing cheapest",
        ),
    ]

    if db.metric===:cosine&&estimates.filter_aware_bound_calibrated
        push!(
            choices,
            (
                strategy = BoundedFilterAwareIVFStrategy(),
                cost = estimates.filter_aware_bound,
                minimum_nprobe = estimates.filter_aware_bound_min_nprobe,
                nprobe = estimates.filter_aware_bound_max_nprobe,
                reason = "measured angular bounds make progressive probing cheapest",
            ),
        )
    end
    selected=choices[findmin([choice.cost for choice in choices])[2]]

    return QueryPlan(
        selected.strategy,
        estimates.selectivity,
        estimates.matching_count,
        selected.minimum_nprobe,
        selected.nprobe,
        selected.cost,
        estimates,
        selected.reason,
    )
end

function strategy_from_symbol(filter::Union{Nothing,FilterExpr}, strategy::Symbol)
    if strategy===:exact
        return filter===nothing ? ExactStrategy() : PreFilterExactStrategy()
    elseif strategy===:ivf
        filter===nothing||throw(ArgumentError("ivf strategy cannot be used with a filter"))
        return IVFStrategy()
    elseif strategy===:prefilter
        filter!==nothing||throw(ArgumentError("prefilter strategy requires a filter"))
        return IVFPreFilterStrategy()
    elseif strategy===:postfilter
        filter!==nothing||throw(ArgumentError("postfilter strategy requires a filter"))
        return IVFPostFilterStrategy()
    elseif strategy===:filter_aware
        filter!==nothing||throw(ArgumentError("filter aware strategy requires a filter"))
        return FilterAwareIVFStrategy()
    elseif strategy===:bound
        filter!==nothing||throw(ArgumentError("bound strategy requires a filter"))
        return BoundedFilterAwareIVFStrategy()
    end

    throw(ArgumentError("unknown search strategy"))
end

"""
    plan_query(db, filter=nothing; k=10, strategy=:auto, config=PlannerConfig())

Return Sen's immutable query plan without executing the search. The plan
includes the selected strategy, estimated candidates and probe range, costs,
and a human-readable selection reason.
"""
function plan_query(
    db::VectorDB,
    filter::Union{Nothing,NamedTuple,FilterExpr};
    k::Int = 10,
    strategy::Symbol = :auto,
    config::PlannerConfig = PlannerConfig(),
)
    filter=normalize_filter(filter)
    k>0||throw(ArgumentError("k must be positive"))
    strategy===:exact&&return exact_query_plan(
        db,
        filter;
        reason = "exact strategy selected manually",
        config = config,
    )
    automatic_plan=choose_strategy(db, filter; k = k, config = config)
    strategy===:auto&&return automatic_plan

    selected_strategy=strategy_from_symbol(filter, strategy)
    selected_name=strategy_name(selected_strategy)
    selected_cost=hasproperty(automatic_plan.costs, selected_name) ?
                  getproperty(automatic_plan.costs, selected_name) :
                  automatic_plan.estimated_cost
    selected_minimum_nprobe=automatic_plan.minimum_nprobe
    selected_nprobe=automatic_plan.nprobe

    if selected_strategy isa ExactStrategy||selected_strategy isa PreFilterExactStrategy
        selected_minimum_nprobe=0
        selected_nprobe=0
    elseif filter!==nothing&&selected_strategy isa IVFPreFilterStrategy
        selected_minimum_nprobe=automatic_plan.costs.prefilter_nprobe
        selected_nprobe=automatic_plan.costs.prefilter_nprobe
    elseif filter!==nothing&&selected_strategy isa IVFPostFilterStrategy
        selected_minimum_nprobe=automatic_plan.costs.postfilter_nprobe
        selected_nprobe=automatic_plan.costs.postfilter_nprobe
    elseif filter!==nothing&&selected_strategy isa FilterAwareIVFStrategy
        selected_minimum_nprobe=automatic_plan.costs.filter_aware_min_nprobe
        selected_nprobe=automatic_plan.costs.filter_aware_max_nprobe
    elseif filter!==nothing&&selected_strategy isa BoundedFilterAwareIVFStrategy
        selected_minimum_nprobe=automatic_plan.costs.filter_aware_bound_min_nprobe
        selected_nprobe=automatic_plan.costs.filter_aware_bound_max_nprobe
    end

    return QueryPlan(
        selected_strategy,
        automatic_plan.selectivity,
        automatic_plan.estimated_candidates,
        selected_minimum_nprobe,
        selected_nprobe,
        selected_cost,
        automatic_plan.costs,
        "strategy selected manually",
    )
end
