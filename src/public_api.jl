const STABLE_API_V1=(
    :VectorDB,
    :SearchResult,
    :PlannerConfig,
    :FilterExpr,
    :Eq,
    :In,
    :Range,
    :And,
    :Or,
    :Not,
    :create_db,
    :insert!,
    :upsert!,
    :update!,
    :delete!,
    :get_record,
    :build!,
    :rebuild!,
    :is_built,
    :is_dirty,
    :search,
    :plan_query,
    :save!,
    :load_db,
    :recover_db,
)

"""
    stable_api()

Return the symbols covered by Sen's stable v0.1 compatibility contract.
Advanced index, planner calibration and benchmark helpers remain available but
may change between minor releases.
"""
function stable_api()
    return STABLE_API_V1
end
