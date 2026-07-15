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
    delta_store::DeltaStore
    base_tombstones::BitVector
    live_count::Int
    database_lock::DatabaseLock
    plan_cache::Dict{Any,Any}
    plan_cache_lock::ReentrantLock
end

function VectorDB(path::String,dim::Int,metric::Symbol,vector_store::VectorStore,metadata_store::MetadataStore,id_store::IDStore,index::Union{Nothing,FilterAwareIVFIndex},filter_index::Union{Nothing,BitsetIndex},build_config::Union{Nothing,IndexBuildConfig},revision::UInt64,index_revision::Union{Nothing,UInt64},database_lock::DatabaseLock,plan_cache::Dict{Any,Any},plan_cache_lock::ReentrantLock,)
    return VectorDB(path,dim,metric,vector_store,metadata_store,id_store,index,filter_index,build_config,revision,index_revision,create_delta_store(dim),falses(length(vector_store)),length(vector_store),database_lock,plan_cache,plan_cache_lock)
end

struct SearchResult
    id
    index::Int
    score::Float32
    metadata::NamedTuple
end

function Base.length(db::VectorDB)
    return with_database_read(db.database_lock) do
        db.live_count
    end
end
