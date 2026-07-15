struct MaintenanceConfig
    enabled::Bool
    minimum_changes::Int
    delta_threshold::Int
    delta_ratio::Float64
    tombstone_threshold::Int
    tombstone_ratio::Float64
    max_retries::Int
    retry_delay_ms::Int
    persist_after_rebuild::Bool
end

function MaintenanceConfig(;enabled::Bool=true,minimum_changes::Int=1_000,delta_threshold::Int=10_000,delta_ratio::Real=0.10,tombstone_threshold::Int=10_000,tombstone_ratio::Real=0.10,max_retries::Int=3,retry_delay_ms::Int=50,persist_after_rebuild::Bool=true,)
    minimum_changes>0||throw(ArgumentError("minimum changes must be positive"))
    delta_threshold>=0||throw(ArgumentError("delta threshold cannot be negative"))
    0.0<=delta_ratio<=1.0||throw(ArgumentError("delta ratio must be between zero and one"))
    tombstone_threshold>=0||throw(ArgumentError("tombstone threshold cannot be negative"))
    0.0<=tombstone_ratio<=1.0||throw(ArgumentError("tombstone ratio must be between zero and one"))
    max_retries>=0||throw(ArgumentError("max retries cannot be negative"))
    retry_delay_ms>=0||throw(ArgumentError("retry delay cannot be negative"))
    return MaintenanceConfig(enabled,minimum_changes,delta_threshold,Float64(delta_ratio),tombstone_threshold,Float64(tombstone_ratio),max_retries,retry_delay_ms,persist_after_rebuild)
end

mutable struct MaintenanceState
    lock::ReentrantLock
    task::Union{Nothing,Task}
    status::Symbol
    pending::Bool
    stop_requested::Bool
    attempts::Int
    last_started_revision::Union{Nothing,UInt64}
    last_completed_revision::Union{Nothing,UInt64}
    last_duration_ms::Float64
    last_error::Union{Nothing,String}
end

function MaintenanceState()
    return MaintenanceState(ReentrantLock(),nothing,:idle,false,false,0,nothing,nothing,0.0,nothing)
end

mutable struct VectorDB
    path::String
    dim::Int
    metric::Symbol
    vector_store::VectorStore
    metadata_store::MetadataStore
    id_store::IDStore
    index::Union{Nothing,FilterAwareIVFIndex}
    filter_index::Union{Nothing,BitsetIndex}
    build_config::Union{Nothing,IndexBuildConfig}
    revision::UInt64
    index_revision::Union{Nothing,UInt64}
    index_bytes::Int
    delta_store::DeltaStore
    base_tombstones::BitVector
    live_count::Int
    wal_revision::Union{Nothing,UInt64}
    wal_checkpoint_revision::Union{Nothing,UInt64}
    checkpoint_operations::Int
    checkpoint_bytes::Int
    checkpoint_retain_snapshots::Int
    writer_lock::Union{Nothing,IOStream}
    closed::Bool
    database_lock::DatabaseLock
    plan_cache::Dict{Any,Any}
    plan_cache_lock::ReentrantLock
    maintenance_config::MaintenanceConfig
    maintenance_state::MaintenanceState
end

function VectorDB(path::String,dim::Int,metric::Symbol,vector_store::VectorStore,metadata_store::MetadataStore,id_store::IDStore,index::Union{Nothing,FilterAwareIVFIndex},filter_index::Union{Nothing,BitsetIndex},build_config::Union{Nothing,IndexBuildConfig},revision::UInt64,index_revision::Union{Nothing,UInt64},database_lock::DatabaseLock,plan_cache::Dict{Any,Any},plan_cache_lock::ReentrantLock;checkpoint_operations::Int=10_000,checkpoint_bytes::Int=64*1024*1024,checkpoint_retain_snapshots::Int=2,maintenance_config::MaintenanceConfig=MaintenanceConfig(),)
    checkpoint_operations>=0||throw(ArgumentError("checkpoint operations cannot be negative"))
    checkpoint_bytes>=0||throw(ArgumentError("checkpoint bytes cannot be negative"))
    checkpoint_retain_snapshots>0||throw(ArgumentError("checkpoint retain snapshots must be positive"))
    index_bytes=index===nothing||filter_index===nothing ? 0 : Base.summarysize((index,filter_index,))
    return VectorDB(path,dim,metric,vector_store,metadata_store,id_store,index,filter_index,build_config,revision,index_revision,index_bytes,create_delta_store(dim),falses(length(vector_store)),length(vector_store),nothing,nothing,checkpoint_operations,checkpoint_bytes,checkpoint_retain_snapshots,nothing,false,database_lock,plan_cache,plan_cache_lock,maintenance_config,MaintenanceState())
end

struct SearchResult
    id
    index::Int
    score::Float32
    metadata::NamedTuple
end

"""A consistent read-only snapshot of a Sen database and its maintenance state."""
struct DatabaseInfo
    path::String
    dim::Int
    metric::Symbol
    live_count::Int
    base_count::Int
    delta_count::Int
    tombstone_count::Int
    delta_ratio::Float64
    tombstone_ratio::Float64
    revision::UInt64
    index_revision::Union{Nothing,UInt64}
    index_count::Int
    index_lists::Int
    index_bytes::Int
    built::Bool
    dirty::Bool
    durable::Bool
    wal_revision::Union{Nothing,UInt64}
    checkpoint_revision::Union{Nothing,UInt64}
    maintenance_enabled::Bool
    maintenance_due::Bool
    maintenance_status::Symbol
    maintenance_running::Bool
    maintenance_attempts::Int
    last_rebuild_revision::Union{Nothing,UInt64}
    last_rebuild_duration_ms::Float64
    last_maintenance_error::Union{Nothing,String}
end

function Base.length(db::VectorDB)
    return with_database_read(db.database_lock) do
        ensure_database_open(db)
        db.live_count
    end
end
