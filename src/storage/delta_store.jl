mutable struct DeltaStore
    vector_store::VectorStore
    metadata_store::MetadataStore
    id_store::IDStore
end

function create_delta_store(dim::Int; initial_capacity::Int = 0)
    return DeltaStore(
        create_vector_store(dim; initial_capacity = initial_capacity),
        create_metadata_store(initial_capacity = initial_capacity),
        create_id_store(initial_capacity = initial_capacity),
    )
end

function Base.length(store::DeltaStore)
    return length(store.vector_store)
end

function has_delta_id(store::DeltaStore, id)
    return has_id(store.id_store, id)
end

function insert_delta!(
    store::DeltaStore,
    vector::AbstractVector{<:Real},
    metadata::NamedTuple,
    id,
)
    vector_position=insert_vector!(store.vector_store, vector)
    metadata_position=insert_metadata!(store.metadata_store, metadata)
    id_position=insert_id!(store.id_store, id)
    vector_position==metadata_position==id_position||error("delta stores are misaligned")
    return id_position
end

function update_delta!(
    store::DeltaStore,
    id;
    vector::Union{Nothing,AbstractVector{<:Real}} = nothing,
    metadata::Union{Nothing,NamedTuple} = nothing,
)
    position=get_position(store.id_store, id)
    vector===nothing||update_vector!(store.vector_store, position, vector)
    metadata===nothing||update_metadata!(store.metadata_store, position, metadata)
    return position
end

function delete_delta!(store::DeltaStore, id)
    position=get_position(store.id_store, id)
    swap_delete_vector!(store.vector_store, position)
    swap_delete_metadata!(store.metadata_store, position)
    deleted=swap_delete_id!(store.id_store, id)
    deleted.position==position||error("delta stores are misaligned")
    return deleted
end

function validate_delta_store(store::DeltaStore)
    count=length(store.vector_store)
    count==length(store.metadata_store)==length(store.id_store)||error(
        "delta stores are misaligned",
    )
    length(store.id_store.positions)==count||error("delta id positions are misaligned")
    return store
end
