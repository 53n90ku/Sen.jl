module Sen

include("concurrency/database_lock.jl")
include("storage/value_codec.jl")
include("storage/vector_store.jl")
include("storage/metadata_store.jl")
include("storage/id_store.jl")
include("storage/delta_store.jl")
include("storage/manifest_store.jl")
include("metrics/dot.jl")
include("metrics/cosine.jl")
include("metrics/topk.jl")
include("filters/filter_expr.jl")
include("filters/bitset_index.jl")
include("indexes/exact.jl")
include("filters/selectivity.jl")
include("indexes/ivf.jl")
include("indexes/filter_aware_ivf.jl")
include("indexes/bound_ivf.jl")
include("storage/index_store.jl")
include("storage/snapshot_store.jl")
include("planner/strategies.jl")
include("types.jl")
include("storage/writer_lock.jl")
include("storage/wal_store.jl")
include("planner/calibration.jl")
include("planner/cost_model.jl")
include("planner/planner.jl")
include("api.jl")
include("public_api.jl")


export VectorDB,SearchResult,PlannerConfig
export FilterExpr,Eq,In,Range,And,Or,Not
export create_db,insert!,upsert!,update!,delete!,get_record
export build!,rebuild!,is_built,is_dirty
export search,plan_query
export save!,load_db,recover_db
export STABLE_API_V1,stable_api

end
