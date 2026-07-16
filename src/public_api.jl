const STABLE_API_V1=(
    :VectorDB,
    :SearchResult,
    :PlannerConfig,
    :DatabaseInfo,
    :MaintenanceConfig,
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
    :compact!,
    :is_built,
    :is_dirty,
    :database_info,
    :configure_maintenance!,
    :maintenance_status,
    :wait_for_maintenance,
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
