function create_db(
    path::AbstractString;
    dim::Int,
    metric::Symbol = :cosine,
    initial_capacity::Int = 0,
    durable::Bool = true,
    checkpoint_operations::Int = 10_000,
    checkpoint_bytes::Int = 64*1024*1024,
    checkpoint_retain_snapshots::Int = 2,
    maintenance_config::MaintenanceConfig = MaintenanceConfig(),
)
    dim>0||throw(ArgumentError("dimension must be positive"))
    metric in (:cosine, :dot)||throw(ArgumentError("metric must be :cosine or :dot"))
    initial_capacity>=0||throw(ArgumentError("initial capacity cannot be negative"))

    vector_store=create_vector_store(dim; initial_capacity = initial_capacity)
    metadata_store=create_metadata_store(initial_capacity = initial_capacity)
    id_store=create_id_store(initial_capacity = initial_capacity)

    db=VectorDB(
        String(path),
        dim,
        metric,
        vector_store,
        metadata_store,
        id_store,
        nothing,
        nothing,
        nothing,
        UInt64(0),
        nothing,
        DatabaseLock(),
        Dict{Any,Any}(),
        ReentrantLock();
        checkpoint_operations = checkpoint_operations,
        checkpoint_bytes = checkpoint_bytes,
        checkpoint_retain_snapshots = checkpoint_retain_snapshots,
        maintenance_config = maintenance_config,
    )

    if durable
        io=acquire_database_writer_lock(path)

        try
            wal_path=database_wal_path(path)
            current_path=database_current_path(path)
            manifest_path=joinpath(path, "manifest.toml")
            any(isfile, (wal_path, current_path, manifest_path))&&throw(
                ArgumentError("database already exists"),
            )
            replace_database_wal(path, db.revision, db.dim, db.metric)
            db.wal_revision=db.revision
            db.wal_checkpoint_revision=db.revision
            attach_database_writer_lock!(db, io)
        catch
            release_database_writer_lock(io)
            rethrow()
        end
    end

    return db
end

function validate_database_fast(db::VectorDB)
    ensure_database_open(db)
    count=length(db.vector_store)
    count==length(db.metadata_store)==length(db.id_store)||error(
        "database stores are misaligned",
    )
    length(db.id_store.positions)==count||error("database id positions are misaligned")
    length(db.base_tombstones)==count||error("database tombstones are misaligned")
    validate_delta_store(db.delta_store)
    db.active_segment.store===db.delta_store||error(
        "active segment does not own the database delta store",
    )
    validate_active_segment_fast(db.active_segment, db.dim, db.revision)

    for segment in db.immutable_segments
        validate_immutable_segment_fast(segment)
    end

    if !isempty(db.immutable_segments)
        primary=first(db.immutable_segments)
        primary.vector_store===db.vector_store||error(
            "primary segment vector store is detached",
        )
        primary.metadata_store===db.metadata_store||error(
            "primary segment metadata store is detached",
        )
        primary.id_store===db.id_store||error("primary segment id store is detached")
        length(primary.excluded)==length(db.base_tombstones)||error(
            "primary segment exclusions are misaligned",
        )
    elseif count>0
        error("database base has no primary segment")
    end

    additional_count=0

    for segment_index = 2:length(db.immutable_segments)
        additional_count+=length(db.immutable_segments[segment_index].vector_store)
    end

    maximum_physical=count+length(db.delta_store)+additional_count
    0<=db.live_count<=maximum_physical||error("database live count is invalid")

    if db.index===nothing
        db.filter_index===nothing||error(
            "database filter index exists without vector index",
        )
        db.index_revision===nothing||error(
            "database index revision exists without vector index",
        )
        db.index_bytes==0||error("database index size exists without vector index")
        db.segment_mode||isempty(db.delta_store.id_store.ids)||error(
            "unindexed legacy database has delta records",
        )
        db.segment_mode||db.live_count==count||error(
            "unindexed legacy database live count is misaligned",
        )
    else
        db.filter_index===nothing&&error(
            "database vector index exists without filter index",
        )
        db.index_revision===nothing&&error("database vector index has no revision")
        db.index_bytes>0||error("database vector index size is invalid")
        db.index_revision<=db.revision||error(
            "database index revision is ahead of database",
        )
        sum(length, db.index.ivf.lists)==count||error("database index count is misaligned")
        db.filter_index.count==count||error("database filter index count is misaligned")
    end

    return db
end

function validate_database(db::VectorDB)
    validate_database_fast(db)
    validate_active_segment(db.active_segment, db.dim, db.revision)

    for segment in db.immutable_segments
        validate_immutable_segment(segment)
    end

    if !isempty(db.mutation_history)
        last(db.mutation_history).revision==db.revision||error(
            "mutation history does not reach the current database revision",
        )

        for index = 2:length(db.mutation_history)
            db.mutation_history[index].revision==db.mutation_history[index-1].revision+UInt64(
                1,
            )||error("mutation history revisions are not contiguous")
        end

        db.segment_mode||db.index_revision===nothing||first(db.mutation_history).revision==db.index_revision+UInt64(
            1,
        )||error("mutation history does not follow the index revision")
    end

    for (position, id) in enumerate(db.id_store.ids)
        get(db.id_store.positions, id, 0)==position||error(
            "database id positions are misaligned",
        )
    end

    for (position, id) in enumerate(db.delta_store.id_store.ids)
        get(db.delta_store.id_store.positions, id, 0)==position||error(
            "delta id positions are misaligned",
        )
        db.segment_mode||has_id(db.id_store, id)&&!db.base_tombstones[get_position(
            db.id_store,
            id,
        )]&&error("database id exists visibly in base and active segment")
    end

    db.live_count==length(database_visible_ids(db))||error(
        "database live count is misaligned",
    )

    return db
end

function register_database_index_build!(db::VectorDB)
    state=db.index_build_state
    lock(state.lock)

    try
        push!(state.revisions, db.revision)
        return db.revision
    finally
        unlock(state.lock)
    end
end

function prune_database_mutation_history_locked!(db::VectorDB)
    state=db.index_build_state
    retention_revision=isempty(state.revisions) ? db.revision : minimum(state.revisions)
    first_retained=findfirst(entry->entry.revision>retention_revision, db.mutation_history)

    if first_retained===nothing
        empty!(db.mutation_history)
    elseif first_retained>1
        deleteat!(db.mutation_history, 1:(first_retained-1))
    end

    return db
end

function prune_database_mutation_history!(db::VectorDB)
    state=db.index_build_state
    lock(state.lock)

    try
        return prune_database_mutation_history_locked!(db)
    finally
        unlock(state.lock)
    end
end

function unregister_database_index_build!(db::VectorDB, revision::UInt64)
    return with_database_write(db.database_lock) do
        state=db.index_build_state
        lock(state.lock)

        try
            position=findfirst(==(revision), state.revisions)
            position===nothing||deleteat!(state.revisions, position)
            prune_database_mutation_history_locked!(db)
            return db
        finally
            unlock(state.lock)
        end
    end
end

function next_database_revision(db::VectorDB)
    db.revision==typemax(UInt64)&&error("database revision overflow")
    return db.revision+UInt64(1)
end

const DATABASE_MUTATION_FAULT_KEY=:sen_database_mutation_fault
const DATABASE_MUTATION_FAULT_STAGES=(:before_wal, :after_wal, :during_apply, :after_apply)

struct InjectedDatabaseMutationError <: Exception
    stage::Symbol
end

struct DatabaseMutationCommittedError <: Exception
    revision::UInt64
    cause::Exception
end

function Base.showerror(io::IO, error::InjectedDatabaseMutationError)
    print(io, "injected database mutation failure at ", error.stage)
end

function Base.showerror(io::IO, error::DatabaseMutationCommittedError)
    print(
        io,
        "database mutation revision ",
        error.revision,
        " committed durably but its in-memory application failed: ",
    )
    showerror(io, error.cause)
end

function inject_database_mutation_fault!(stage::Symbol)
    stage in DATABASE_MUTATION_FAULT_STAGES||throw(
        ArgumentError("unsupported database mutation fault stage"),
    )
    task_local_storage()[DATABASE_MUTATION_FAULT_KEY]=stage
    return stage
end

function clear_database_mutation_fault!()
    storage=task_local_storage()
    haskey(storage, DATABASE_MUTATION_FAULT_KEY)&&delete!(
        storage,
        DATABASE_MUTATION_FAULT_KEY,
    )
    return nothing
end

function database_mutation_fault_stage()
    return get(task_local_storage(), DATABASE_MUTATION_FAULT_KEY, nothing)
end

function maybe_inject_database_mutation_fault!(stage::Symbol)
    database_mutation_fault_stage()===stage||return nothing
    clear_database_mutation_fault!()
    throw(InjectedDatabaseMutationError(stage))
end

function clear_plan_cache!(db::VectorDB)
    lock(db.plan_cache_lock)

    try
        empty!(db.plan_cache)
    finally
        unlock(db.plan_cache_lock)
    end

    return db
end

function has_usable_base(db::VectorDB)
    return db.index!==nothing&&db.filter_index!==nothing&&db.index_revision!==nothing
end

function synchronize_primary_segment!(db::VectorDB; revision_end::UInt64 = db.revision)
    revision_start=isempty(db.immutable_segments) ? UInt64(0) :
                   first(db.immutable_segments).revision_start
    id=isempty(db.immutable_segments) ? "base-$(revision_end)" :
       first(db.immutable_segments).id
    primary=create_immutable_segment(
        id,
        revision_start,
        revision_end,
        db.vector_store,
        db.metadata_store,
        db.id_store;
        excluded = copy(db.base_tombstones),
        index = db.index,
        filter_index = db.filter_index,
        build_config = db.build_config,
        index_bytes = db.index_bytes,
    )

    if isempty(db.immutable_segments)
        push!(db.immutable_segments, primary)
    else
        db.immutable_segments[1]=primary
    end

    return db
end

function synchronize_active_segment!(db::VectorDB)
    db.active_segment.store=db.delta_store
    return db
end

function database_visible_ids(db::VectorDB)
    return segment_topology_visible_ids(db.immutable_segments, db.active_segment)
end

function database_segment_exclusions(db::VectorDB)
    exclusions=[copy(segment.excluded) for segment in db.immutable_segments]
    seen=Set{Any}(db.active_segment.tombstone_ids)
    union!(seen, db.delta_store.id_store.ids)

    for segment_index in reverse(eachindex(db.immutable_segments))
        segment=db.immutable_segments[segment_index]
        union!(seen, segment.tombstone_ids)
        excluded=exclusions[segment_index]

        for (position, id) in enumerate(segment.id_store.ids)
            if id in seen
                excluded[position]=true
            else
                push!(seen, id)
            end
        end
    end

    return exclusions
end

function cached_database_segment_exclusions(db::VectorDB)
    key=(
        :segment_exclusions,
        db.revision,
        length(db.immutable_segments),
        db.active_segment.id,
    )
    lock(db.plan_cache_lock)

    try
        return get!(db.plan_cache, key) do
            database_segment_exclusions(db)
        end
    finally
        unlock(db.plan_cache_lock)
    end
end

function database_record_location(db::VectorDB, id)
    if id in db.active_segment.tombstone_ids
        return nothing
    end

    if has_delta_id(db.delta_store, id)
        return (
            kind = :active,
            segment_index = 0,
            position = get_position(db.delta_store.id_store, id),
        )
    end

    for segment_index in reverse(eachindex(db.immutable_segments))
        segment=db.immutable_segments[segment_index]
        id in segment.tombstone_ids&&return nothing

        if has_id(segment.id_store, id)
            position=get_position(segment.id_store, id)
            segment.excluded[position]&&return nothing
            return (kind = :immutable, segment_index = segment_index, position = position)
        end
    end

    return nothing
end

function enable_database_segment_mode!(db::VectorDB)
    db.segment_mode||synchronize_primary_segment!(db)
    synchronize_active_segment!(db)
    db.segment_mode=true
    return db
end

function seal_active_segment_locked!(db::VectorDB)
    enable_database_segment_mode!(db)
    active=db.active_segment
    active_segment_is_empty(active)&&return nothing
    active.revision_start<=active.revision_end||error(
        "cannot seal an active segment without a revision range",
    )
    segment=create_immutable_segment(
        "segment-$(active.revision_start)-$(active.revision_end)-$(time_ns())",
        active.revision_start,
        active.revision_end,
        active.store.vector_store,
        active.store.metadata_store,
        active.store.id_store;
        tombstone_ids = copy(active.tombstone_ids),
        filter_index = build_bitset_index(stored_metadata(active.store.metadata_store)),
    )
    push!(db.immutable_segments, segment)
    reset_active_segment!(active, db.dim, db.revision)
    db.delta_store=active.store
    clear_plan_cache!(db)
    validate_database(db)
    return segment
end

function seal_active_segment!(db::VectorDB)
    return with_database_write(db.database_lock) do
        validate_database(db)
        segment=seal_active_segment_locked!(db)
        segment===nothing||schedule_segment_indexing_locked!(db)
        return segment
    end
end

function logical_length(db::VectorDB)
    return db.live_count
end

function delta_search_work(db::VectorDB)
    return length(db.delta_store)
end

function active_put_growth(db::VectorDB, ids::AbstractVector)
    return count(ids) do id
        !has_delta_id(db.delta_store, id)&&!(id in db.active_segment.tombstone_ids)
    end
end

function active_delete_growth(db::VectorDB, ids::AbstractVector)
    return count(ids) do id
        location=database_record_location(db, id)
        location!==nothing&&location.kind===:immutable
    end
end

function seal_and_schedule_active_segment_locked!(db::VectorDB)
    segment=seal_active_segment_locked!(db)
    segment===nothing||schedule_segment_indexing_locked!(db)
    return segment
end

function ensure_active_segment_capacity_locked!(
    db::VectorDB,
    ids::AbstractVector,
    operation::Symbol,
)
    operation in (:put, :delete)||throw(
        ArgumentError("unsupported active segment operation"),
    )
    limit=db.maintenance_config.active_segment_threshold
    growth=operation===:put ? active_put_growth(db, ids) : active_delete_growth(db, ids)
    active_segment_work(db.active_segment)+growth<=limit&&return db
    length(ids)<=limit||throw(
        ArgumentError("mutation would exceed the configured active segment threshold"),
    )
    active_segment_is_empty(db.active_segment)||seal_and_schedule_active_segment_locked!(db)
    post_seal_growth=length(ids)
    post_seal_growth<=limit||throw(
        ArgumentError("mutation would exceed the configured active segment threshold"),
    )
    return db
end

function maybe_seal_active_segment_locked!(db::VectorDB)
    db.segment_mode||return nothing
    active_segment_work(db.active_segment)>=db.maintenance_config.active_segment_threshold||return nothing
    return seal_and_schedule_active_segment_locked!(db)
end

function ensure_delta_search_capacity!(db::VectorDB, ids::AbstractVector)
    if db.segment_mode
        ensure_active_segment_capacity_locked!(db, ids, :put)
        return db
    end

    has_usable_base(db)||return db
    limit=db.maintenance_config.max_delta_search_records
    additions=count(id->!has_delta_id(db.delta_store, id), ids)
    delta_search_work(db)+additions<=limit&&return db
    length(ids)<=limit||throw(
        ArgumentError("mutation would exceed the configured delta search bound"),
    )
    rebuild_database_maintenance!(db)
    isempty(db.delta_store.id_store.ids)||error(
        "foreground delta compaction did not clear the mutation tail",
    )
    length(ids)<=limit||error(
        "mutation exceeds the configured delta search bound after compaction",
    )
    return db
end

function has_database_id(db::VectorDB, id)
    return database_record_location(db, id)!==nothing
end

function next_database_id(db::VectorDB)
    id=logical_length(db)+1

    while has_database_id(db, id)
        id+=1
    end

    return id
end

function tombstone_base_id!(db::VectorDB, id)
    position=get_position(db.id_store, id)
    db.base_tombstones[position]=true
    return position
end

function finish_database_mutation!(db::VectorDB, revision::UInt64, wal_body::Vector{UInt8})
    revision==next_database_revision(db)||error("database mutation revision is invalid")
    db.revision=revision
    if db.segment_mode
        mark_active_segment_revision!(db.active_segment, revision)
    else
        synchronize_primary_segment!(db; revision_end = revision)
    end
    record_database_mutation!(db, revision, wal_body)
    clear_plan_cache!(db)
    maybe_seal_active_segment_locked!(db)
    validate_database_fast(db)
    maybe_checkpoint_database!(db)
    maybe_schedule_database_maintenance!(db)
    return db
end

function database_mutation_state(db::VectorDB)
    return deepcopy((
        vector_store = db.vector_store,
        metadata_store = db.metadata_store,
        id_store = db.id_store,
        index = db.index,
        filter_index = db.filter_index,
        build_config = db.build_config,
        revision = db.revision,
        index_revision = db.index_revision,
        index_bytes = db.index_bytes,
        delta_store = db.delta_store,
        base_tombstones = db.base_tombstones,
        immutable_segments = db.immutable_segments,
        active_segment = db.active_segment,
        segment_mode = db.segment_mode,
        live_count = db.live_count,
        mutation_history = db.mutation_history,
        wal_revision = db.wal_revision,
        wal_checkpoint_revision = db.wal_checkpoint_revision,
    ))
end

function install_database_mutation_state!(db::VectorDB, state)
    old_vector_store=db.vector_store
    db.vector_store=state.vector_store
    db.metadata_store=state.metadata_store
    db.id_store=state.id_store
    db.index=state.index
    db.filter_index=state.filter_index
    db.build_config=state.build_config
    db.revision=state.revision
    db.index_revision=state.index_revision
    db.index_bytes=state.index_bytes
    db.delta_store=state.delta_store
    db.base_tombstones=state.base_tombstones
    db.immutable_segments=state.immutable_segments
    db.active_segment=state.active_segment
    db.segment_mode=state.segment_mode
    db.live_count=state.live_count
    db.mutation_history=state.mutation_history
    db.wal_revision=state.wal_revision
    db.wal_checkpoint_revision=state.wal_checkpoint_revision
    old_vector_store===db.vector_store||release_vector_store_mapping!(old_vector_store)
    clear_plan_cache!(db)
    validate_database(db)
    return db
end

function recover_committed_database_mutation!(
    db::VectorDB,
    revision::UInt64,
    wal_body::Vector{UInt8},
)
    retained_history=copy(db.mutation_history)
    recovered=load_db_without_writer_lock(
        db.path;
        rebuild = false,
        recover = true,
        checkpoint_operations = db.checkpoint_operations,
        checkpoint_bytes = db.checkpoint_bytes,
        checkpoint_retain_snapshots = db.checkpoint_retain_snapshots,
        maintenance_config = db.maintenance_config,
        mmap_vectors = false,
    )
    install_database_mutation_state!(db, database_mutation_state(recovered))
    state=db.index_build_state
    lock(state.lock)

    try
        if !isempty(state.revisions)
            retention_revision=minimum(state.revisions)
            history=DatabaseMutationEntry[
                entry for entry in retained_history if entry.revision>retention_revision
            ]

            if revision>retention_revision&&!any(entry->entry.revision==revision, history)
                push!(history, DatabaseMutationEntry(revision, copy(wal_body)))
            end

            sort!(history; by = entry->entry.revision)
            db.mutation_history=history
        end
    finally
        unlock(state.lock)
    end

    validate_database(db)
    return db
end

function commit_database_mutation!(
    apply::Function,
    db::VectorDB,
    revision::UInt64,
    wal_body::Vector{UInt8},
)
    maybe_inject_database_mutation_fault!(:before_wal)
    durable=append_database_wal_body_if_active!(db, revision, wal_body)
    fault_stage=database_mutation_fault_stage()
    backup=!durable&&fault_stage in (:during_apply, :after_apply) ?
           database_mutation_state(db) : nothing

    try
        maybe_inject_database_mutation_fault!(:after_wal)
        result=apply()
        maybe_inject_database_mutation_fault!(:after_apply)
        finish_database_mutation!(db, revision, wal_body)
        return result
    catch error
        if durable
            recover_committed_database_mutation!(db, revision, wal_body)
            throw(DatabaseMutationCommittedError(revision, error))
        elseif backup!==nothing
            install_database_mutation_state!(db, backup)
        end

        rethrow()
    end
end

function is_built(db::VectorDB)
    return with_database_read(db.database_lock) do
        has_usable_base(db)&&db.index_revision==db.revision
    end
end

function is_dirty(db::VectorDB)
    return with_database_read(db.database_lock) do
        db.revision>0&&(db.index_revision===nothing||db.index_revision!=db.revision)
    end
end

function Base.insert!(
    db::VectorDB,
    vector::AbstractVector{<:Real},
    metadata::NamedTuple;
    id = nothing,
)
    return with_database_write(db.database_lock) do
        validate_database_fast(db)
        converted=convert_validated_vector(vector, db.dim, db.metric)
        resolved_id=id===nothing ? next_database_id(db) : id
        resolved_id===nothing&&throw(ArgumentError("id cannot be nothing"))
        has_database_id(db, resolved_id)&&throw(ArgumentError("id already exists"))
        ensure_delta_search_capacity!(db, [resolved_id])
        revision=next_database_revision(db)
        wal_body=database_wal_put_body(revision, [converted], [metadata], [resolved_id])
        db.segment_mode||has_usable_base(db)||make_vector_store_writable!(db.vector_store)

        return commit_database_mutation!(db, revision, wal_body) do
            insert_database_record!(db, converted, metadata, resolved_id)
            maybe_inject_database_mutation_fault!(:during_apply)
            db.live_count+=1
            return resolved_id
        end
    end
end

function prepare_database_batch(
    db::VectorDB,
    vectors::AbstractMatrix,
    metadata::AbstractVector{<:NamedTuple},
    ids;
    allow_existing::Bool,
)
    _, count=size(vectors)
    length(metadata)==count||throw(
        DimensionMismatch("metadata count doesnt match vector count"),
    )
    converted=convert_validated_vectors(vectors, db.dim, db.metric)
    converted_metadata=NamedTuple[value for value in metadata]
    resolved_ids=if ids===nothing
        generated=Vector{Any}(undef, count)
        reserved=Set{Any}()
        candidate=next_database_id(db)

        for index = 1:count
            while has_database_id(db, candidate)||candidate in reserved
                candidate+=1
            end

            generated[index]=candidate
            push!(reserved, candidate)
            candidate+=1
        end

        generated
    else
        collect(ids)
    end

    length(resolved_ids)==count||throw(
        DimensionMismatch("id count doesnt match vector count"),
    )
    all(id->id!==nothing, resolved_ids)||throw(ArgumentError("id cannot be nothing"))
    length(Set{Any}(resolved_ids))==count||throw(ArgumentError("batch ids must be unique"))

    if !allow_existing
        any(id->has_database_id(db, id), resolved_ids)&&throw(
            ArgumentError("id already exists"),
        )
    end

    return (
        vectors = converted,
        metadata = converted_metadata,
        ids = resolved_ids,
        count = count,
    )
end

function insert_database_record!(
    db::VectorDB,
    vector::AbstractVector{<:Real},
    metadata::NamedTuple,
    id,
)
    if db.segment_mode||has_usable_base(db)
        delete!(db.active_segment.tombstone_ids, id)
        insert_delta!(db.delta_store, vector, metadata, id)
    else
        make_vector_store_writable!(db.vector_store)
        vector_position=insert_vector!(db.vector_store, vector)
        metadata_position=insert_metadata!(db.metadata_store, metadata)
        id_position=insert_id!(db.id_store, id)
        vector_position==metadata_position==id_position||error(
            "database stores are misaligned",
        )
        push!(db.base_tombstones, false)
    end

    return id
end

function update_database_record!(
    db::VectorDB,
    id,
    vector::AbstractVector{<:Real},
    metadata::NamedTuple,
)
    location=database_record_location(db, id)

    if location!==nothing&&location.kind===:active
        update_delta!(db.delta_store, id; vector = vector, metadata = metadata)
    elseif location!==nothing&&location.kind===:immutable
        if !db.segment_mode&&location.segment_index==1
            make_vector_store_writable!(db.vector_store)
            update_vector!(db.vector_store, location.position, vector)
            update_metadata!(db.metadata_store, location.position, metadata)
        else
            location.segment_index==1&&tombstone_base_id!(db, id)
            delete!(db.active_segment.tombstone_ids, id)
            insert_delta!(db.delta_store, vector, metadata, id)
        end
    else
        insert_database_record!(db, vector, metadata, id)
    end

    return id
end

function swap_delete_tombstone!(tombstones::BitVector, position::Int)
    last_position=length(tombstones)
    position==last_position||(tombstones[position]=tombstones[last_position])
    pop!(tombstones)
    return tombstones
end

function delete_database_record!(db::VectorDB, id)
    location=database_record_location(db, id)
    location===nothing&&throw(KeyError(id))

    if location.kind===:active
        delete_delta!(db.delta_store, id)
        db.segment_mode&&push!(db.active_segment.tombstone_ids, id)
        return db
    end

    if db.segment_mode
        location.segment_index==1&&(db.base_tombstones[location.position]=true)
        push!(db.active_segment.tombstone_ids, id)
    else
        make_vector_store_writable!(db.vector_store)
        swap_delete_vector!(db.vector_store, location.position)
        swap_delete_metadata!(db.metadata_store, location.position)
        deleted=swap_delete_id!(db.id_store, id)
        deleted.position==location.position||error("database stores are misaligned")
        swap_delete_tombstone!(db.base_tombstones, location.position)
    end

    return db
end

function Base.insert!(
    db::VectorDB,
    vectors::AbstractMatrix,
    metadata::AbstractVector{<:NamedTuple};
    ids = nothing,
)
    return with_database_write(db.database_lock) do
        validate_database_fast(db)
        batch=prepare_database_batch(db, vectors, metadata, ids; allow_existing = false)
        batch.count==0&&return batch.ids
        ensure_delta_search_capacity!(db, batch.ids)
        revision=next_database_revision(db)
        wal_vectors=[@view batch.vectors[:, index] for index = 1:batch.count]
        wal_body=database_wal_put_body(revision, wal_vectors, batch.metadata, batch.ids)
        db.segment_mode||has_usable_base(db)||make_vector_store_writable!(db.vector_store)

        return commit_database_mutation!(db, revision, wal_body) do
            for index = 1:batch.count
                insert_database_record!(
                    db,
                    @view(batch.vectors[:, index]),
                    batch.metadata[index],
                    batch.ids[index],
                )
                maybe_inject_database_mutation_fault!(:during_apply)
            end

            db.live_count+=batch.count
            return batch.ids
        end
    end
end

function upsert!(db::VectorDB, vector::AbstractVector{<:Real}, metadata::NamedTuple; id)
    return with_database_write(db.database_lock) do
        validate_database_fast(db)
        id===nothing&&throw(ArgumentError("id cannot be nothing"))
        converted=convert_validated_vector(vector, db.dim, db.metric)
        existing=has_database_id(db, id)
        ensure_delta_search_capacity!(db, [id])
        revision=next_database_revision(db)
        wal_body=database_wal_put_body(revision, [converted], [metadata], [id])
        db.segment_mode||has_usable_base(db)||make_vector_store_writable!(db.vector_store)

        return commit_database_mutation!(db, revision, wal_body) do
            update_database_record!(db, id, converted, metadata)
            maybe_inject_database_mutation_fault!(:during_apply)
            existing||(db.live_count+=1)
            return id
        end
    end
end

function upsert!(
    db::VectorDB,
    vectors::AbstractMatrix,
    metadata::AbstractVector{<:NamedTuple};
    ids,
)
    return with_database_write(db.database_lock) do
        validate_database_fast(db)
        batch=prepare_database_batch(db, vectors, metadata, ids; allow_existing = true)
        batch.count==0&&return batch.ids
        new_count=count(id->!has_database_id(db, id), batch.ids)
        ensure_delta_search_capacity!(db, batch.ids)
        revision=next_database_revision(db)
        wal_vectors=[@view batch.vectors[:, index] for index = 1:batch.count]
        wal_body=database_wal_put_body(revision, wal_vectors, batch.metadata, batch.ids)
        db.segment_mode||has_usable_base(db)||make_vector_store_writable!(db.vector_store)

        return commit_database_mutation!(db, revision, wal_body) do
            for index = 1:batch.count
                id=batch.ids[index]
                update_database_record!(
                    db,
                    id,
                    @view(batch.vectors[:, index]),
                    batch.metadata[index],
                )
                maybe_inject_database_mutation_fault!(:during_apply)
            end

            db.live_count+=new_count
            return batch.ids
        end
    end
end

function update!(
    db::VectorDB,
    id;
    vector::Union{Nothing,AbstractVector{<:Real}} = nothing,
    metadata::Union{Nothing,NamedTuple} = nothing,
)
    return with_database_write(db.database_lock) do
        validate_database_fast(db)
        vector===nothing&&metadata===nothing&&throw(
            ArgumentError("update requires a vector or metadata"),
        )
        has_database_id(db, id)||throw(KeyError(id))
        converted=if vector===nothing
            nothing
        else
            convert_validated_vector(vector, db.dim, db.metric)
        end

        current=get_database_record(db, id)
        resolved_vector=converted===nothing ? current.vector : converted
        resolved_metadata=metadata===nothing ? current.metadata : metadata
        ensure_delta_search_capacity!(db, [id])
        revision=next_database_revision(db)
        wal_body=database_wal_put_body(
            revision,
            [resolved_vector],
            [resolved_metadata],
            [id],
        )
        db.segment_mode||has_usable_base(db)||make_vector_store_writable!(db.vector_store)

        return commit_database_mutation!(db, revision, wal_body) do
            update_database_record!(db, id, resolved_vector, resolved_metadata)
            maybe_inject_database_mutation_fault!(:during_apply)
            return db
        end
    end
end

function Base.delete!(db::VectorDB, id)
    return with_database_write(db.database_lock) do
        validate_database_fast(db)
        has_database_id(db, id)||throw(KeyError(id))
        db.segment_mode&&ensure_active_segment_capacity_locked!(db, [id], :delete)
        revision=next_database_revision(db)
        wal_body=database_wal_delete_body(revision, [id])
        db.segment_mode||has_usable_base(db)||make_vector_store_writable!(db.vector_store)

        return commit_database_mutation!(db, revision, wal_body) do
            delete_database_record!(db, id)
            maybe_inject_database_mutation_fault!(:during_apply)
            db.live_count-=1
            return db
        end
    end
end

function Base.delete!(db::VectorDB, ids::AbstractVector)
    return with_database_write(db.database_lock) do
        validate_database_fast(db)
        resolved_ids=collect(ids)
        isempty(resolved_ids)&&return db
        length(Set{Any}(resolved_ids))==length(resolved_ids)||throw(
            ArgumentError("batch ids must be unique"),
        )

        for id in resolved_ids
            has_database_id(db, id)||throw(KeyError(id))
        end


        db.segment_mode&&ensure_active_segment_capacity_locked!(db, resolved_ids, :delete)

        revision=next_database_revision(db)
        wal_body=database_wal_delete_body(revision, resolved_ids)
        db.segment_mode||has_usable_base(db)||make_vector_store_writable!(db.vector_store)

        return commit_database_mutation!(db, revision, wal_body) do
            for id in resolved_ids
                delete_database_record!(db, id)
                maybe_inject_database_mutation_fault!(:during_apply)
            end

            db.live_count-=length(resolved_ids)
            return db
        end
    end
end

function get_database_record(db::VectorDB, id)
    location=database_record_location(db, id)
    location===nothing&&throw(KeyError(id))

    if location.kind===:active
        position=location.position
        return (
            id = get_id(db.delta_store.id_store, position),
            vector = collect(get_vector(db.delta_store.vector_store, position)),
            metadata = get_metadata(db.delta_store.metadata_store, position),
        )
    end

    segment=db.immutable_segments[location.segment_index]
    position=location.position
    return (
        id = get_id(segment.id_store, position),
        vector = collect(get_vector(segment.vector_store, position)),
        metadata = get_metadata(segment.metadata_store, position),
    )
end

function get_record(db::VectorDB, id)
    return with_database_read(db.database_lock) do
        validate_database_fast(db)
        return get_database_record(db, id)
    end
end

function materialize_database(db::VectorDB)
    count=logical_length(db)
    vector_store=create_vector_store(db.dim; initial_capacity = count)
    metadata_store=create_metadata_store(initial_capacity = count)
    id_store=create_id_store(initial_capacity = count)

    exclusions=database_segment_exclusions(db)

    for (segment_index, segment) in enumerate(db.immutable_segments)
        excluded=exclusions[segment_index]

        for position = 1:length(segment.vector_store)
            excluded[position]&&continue
            insert_vector!(vector_store, get_vector(segment.vector_store, position))
            insert_metadata!(metadata_store, get_metadata(segment.metadata_store, position))
            insert_id!(id_store, get_id(segment.id_store, position))
        end
    end

    for position = 1:length(db.delta_store)
        insert_vector!(vector_store, get_vector(db.delta_store.vector_store, position))
        insert_metadata!(
            metadata_store,
            get_metadata(db.delta_store.metadata_store, position),
        )
        insert_id!(id_store, get_id(db.delta_store.id_store, position))
    end

    length(vector_store)==count||error("materialized database count is misaligned")
    return (
        vector_store = vector_store,
        metadata_store = metadata_store,
        id_store = id_store,
    )
end

function build!(
    db::VectorDB;
    nlists::Int,
    iterations::Int = 20,
    seed::Int = 42,
    restarts::Int = 1,
    training_count::Union{Nothing,Int} = nothing,
    _snapshot_hook::Union{Nothing,Function} = nothing,
)
    build_revision=nothing

    try
        snapshot=with_database_read(db.database_lock) do
            validate_database_fast(db)
            count=logical_length(db)
            count>0||throw(ArgumentError("database cannot be empty"))
            resolved_training_count=training_count===nothing ? count : training_count
            config=IndexBuildConfig(
                nlists,
                count;
                iterations = iterations,
                seed = seed,
                restarts = restarts,
                training_count = resolved_training_count,
            )
            build_revision=register_database_index_build!(db)
            stores=materialize_database(db)
            return (
                revision = build_revision,
                config = config,
                stores = stores,
                metric = db.metric,
            )
        end

        _snapshot_hook===nothing||_snapshot_hook(snapshot)

        vectors=stored_vectors(snapshot.stores.vector_store)
        metadata=stored_metadata(snapshot.stores.metadata_store)
        index=build_filter_aware_ivf(
            vectors,
            metadata;
            nlists = snapshot.config.nlists,
            iterations = snapshot.config.iterations,
            seed = snapshot.config.seed,
            metric = snapshot.metric,
            restarts = snapshot.config.restarts,
            training_count = snapshot.config.training_count,
        )
        filter_index=build_bitset_index(metadata)
        index_bytes=database_index_bytes(index, filter_index)
        built=(
            revision = snapshot.revision,
            config = snapshot.config,
            stores = snapshot.stores,
            index = index,
            filter_index = filter_index,
            index_bytes = index_bytes,
        )

        return install_database_index!(db, built)
    finally
        build_revision===nothing||unregister_database_index_build!(db, build_revision)
    end
end

function database_mutation_tail(db::VectorDB, revision::UInt64)
    revision<=db.revision||throw(ArgumentError("index build revision is ahead of database"))
    entries=DatabaseMutationEntry[
        entry for entry in db.mutation_history if entry.revision>revision
    ]
    expected=revision

    for entry in entries
        expected==typemax(UInt64)&&throw(
            ArgumentError("mutation history revision overflow"),
        )
        expected+=UInt64(1)
        entry.revision==expected||throw(
            ArgumentError("mutation history does not cover the index build tail"),
        )
    end

    expected==db.revision||throw(
        ArgumentError("mutation history does not reach the current database revision"),
    )
    return entries
end

function rebase_database_index(db::VectorDB, built, entries::Vector{DatabaseMutationEntry})
    candidate=VectorDB(
        db.path,
        db.dim,
        db.metric,
        built.stores.vector_store,
        built.stores.metadata_store,
        built.stores.id_store,
        built.index,
        built.filter_index,
        built.config,
        built.revision,
        built.revision,
        DatabaseLock(),
        Dict{Any,Any}(),
        ReentrantLock();
        checkpoint_operations = db.checkpoint_operations,
        checkpoint_bytes = db.checkpoint_bytes,
        checkpoint_retain_snapshots = db.checkpoint_retain_snapshots,
        maintenance_config = db.maintenance_config,
    )
    candidate.index_bytes=built.index_bytes

    for entry in entries
        apply_database_wal_record!(candidate, read_database_wal_record(entry.body, db.dim))
    end

    candidate.revision==db.revision||error(
        "rebased index generation did not reach the current revision",
    )
    candidate.live_count==db.live_count||error(
        "rebased index generation changed the logical record count",
    )
    delta_search_work(candidate)<=db.maintenance_config.max_delta_search_records||throw(
        ArgumentError("rebased index generation exceeds the configured delta search bound"),
    )
    validate_database(candidate)
    return candidate
end

function install_database_index!(db::VectorDB, built)
    return with_database_write(db.database_lock) do
        all(
            property->hasproperty(built, property),
            (:revision, :config, :stores, :index, :filter_index, :index_bytes),
        )||throw(ArgumentError("index build generation is incomplete"))
        built.revision<=db.revision||throw(
            ArgumentError("index build revision is ahead of database"),
        )
        db.index_revision!==nothing&&built.revision<db.index_revision&&return db
        entries=database_mutation_tail(db, built.revision)
        candidate=rebase_database_index(db, built, entries)
        old_vector_store=db.vector_store
        db.vector_store=candidate.vector_store
        db.metadata_store=candidate.metadata_store
        db.id_store=candidate.id_store
        db.index=candidate.index
        db.filter_index=candidate.filter_index
        db.build_config=candidate.build_config
        db.index_revision=candidate.index_revision
        db.index_bytes=candidate.index_bytes
        db.delta_store=candidate.delta_store
        db.base_tombstones=candidate.base_tombstones
        db.immutable_segments=candidate.immutable_segments
        db.active_segment=candidate.active_segment
        db.segment_mode=true
        db.live_count=candidate.live_count
        db.mutation_history=candidate.mutation_history
        old_vector_store===db.vector_store||release_vector_store_mapping!(old_vector_store)
        clear_plan_cache!(db)

        validate_database(db)
        return db
    end
end

function rebuild!(db::VectorDB)
    config=with_database_read(db.database_lock) do
        db.build_config===nothing&&throw(
            ArgumentError("database has no previous index build configuration"),
        )
        return db.build_config
    end
    count=with_database_read(db.database_lock) do
        logical_length(db)
    end
    config.nlists<=count||throw(
        ArgumentError("database has fewer vectors than its configured list count"),
    )
    training_count=min(count, max(config.nlists, config.training_count))
    return build!(
        db;
        nlists = config.nlists,
        iterations = config.iterations,
        seed = config.seed,
        restarts = config.restarts,
        training_count = training_count,
    )
end

"""Compact all immutable and active segments into a freshly indexed base generation."""
function compact!(db::VectorDB)
    return rebuild!(db)
end

function database_search_results(
    raw_results::AbstractVector,
    id_store::IDStore;
    index_offset::Int = 0,
)
    results=Vector{SearchResult}(undef, length(raw_results))

    for index in eachindex(raw_results)
        result=raw_results[index]
        results[index]=SearchResult(
            get_id(id_store, result.index),
            index_offset+result.index,
            result.score,
            result.metadata,
        )
    end

    return results
end

function merge_database_search_results(
    base_results::Vector{SearchResult},
    delta_results::Vector{SearchResult},
    k::Int,
)
    isempty(delta_results)&&return base_results
    isempty(base_results)&&return delta_results
    count=min(k, length(base_results)+length(delta_results))
    results=Vector{SearchResult}(undef, count)
    base_position=1
    delta_position=1

    for position = 1:count
        take_base=if base_position>length(base_results)
            false
        elseif delta_position>length(delta_results)
            true
        else
            base_score=base_results[base_position].score
            delta_score=delta_results[delta_position].score
            isless(delta_score, base_score)||isequal(base_score, delta_score)
        end

        if take_base
            results[position]=base_results[base_position]
            base_position+=1
        else
            results[position]=delta_results[delta_position]
            delta_position+=1
        end
    end

    return results
end

function search_immutable_segment_raw(
    segment::ImmutableSegment,
    query::AbstractVector{<:Real},
    plan;
    k::Int,
    nprobe::Union{Nothing,Int},
    filter::Union{Nothing,FilterExpr},
    postfilter_oversample::Int,
    adaptive_postfilter::Bool,
    adaptive::Bool,
    max_nprobe::Union{Nothing,Int},
    candidate_multiplier::Float64,
    postfilter_candidate_multiplier::Float64,
    vector_weight::Float64,
    filter_weight::Float64,
    rerank_factor::Int,
    metric::Symbol,
    excluded::BitVector,
)
    vectors=stored_vectors(segment.vector_store)
    metadata=stored_metadata(segment.metadata_store)
    index=segment.index

    if index===nothing||plan===nothing||plan.strategy isa ExactStrategy||plan.strategy isa
                                                                         PreFilterExactStrategy
        vector_norms=index===nothing ? nothing : index.ivf.vector_norms
        return search_exact(
            vectors,
            metadata,
            query;
            k = k,
            metric = metric,
            filter = filter,
            filter_index = segment.filter_index,
            vector_norms = vector_norms,
            excluded = excluded,
        )
    end

    list_count=length(index.ivf.lists)
    resolved_nprobe=min(list_count, nprobe===nothing ? max(1, plan.nprobe) : nprobe)
    resolved_minimum_nprobe=min(
        list_count,
        nprobe===nothing ? max(1, plan.minimum_nprobe) : nprobe,
    )
    resolved_max_nprobe=max_nprobe===nothing ? resolved_nprobe :
                        clamp(max_nprobe, resolved_nprobe, list_count)

    if plan.strategy isa IVFStrategy
        return search_ivf(
            index.ivf,
            vectors,
            metadata,
            query;
            k = k,
            nprobe = resolved_nprobe,
            metric = metric,
            excluded = excluded,
        )
    elseif plan.strategy isa IVFPreFilterStrategy
        return search_ivf_prefilter(
            index,
            vectors,
            metadata,
            query;
            k = k,
            nprobe = resolved_nprobe,
            metric = metric,
            filter = filter,
            excluded = excluded,
        )
    elseif plan.strategy isa IVFPostFilterStrategy
        resolved_oversample=adaptive_postfilter ?
                            resolve_postfilter_oversample(
            postfilter_oversample,
            plan.selectivity;
            candidate_multiplier = postfilter_candidate_multiplier,
        ) : postfilter_oversample
        return search_ivf_postfilter(
            index.ivf,
            vectors,
            metadata,
            query;
            k = k,
            nprobe = resolved_nprobe,
            metric = metric,
            filter = filter,
            oversample = resolved_oversample,
            excluded = excluded,
        )
    elseif plan.strategy isa BoundedFilterAwareIVFStrategy
        return search_filter_aware_bound(
            index,
            vectors,
            metadata,
            query;
            k = k,
            minimum_nprobe = resolved_minimum_nprobe,
            max_nprobe = resolved_max_nprobe,
            metric = metric,
            filter = filter,
            excluded = excluded,
        )
    end

    minimum_nprobe=adaptive ? resolved_minimum_nprobe : resolved_nprobe
    return search_filter_aware_ivf(
        index,
        vectors,
        metadata,
        query;
        k = k,
        nprobe = minimum_nprobe,
        metric = metric,
        filter = filter,
        adaptive = adaptive,
        max_nprobe = resolved_max_nprobe,
        candidate_multiplier = candidate_multiplier,
        vector_weight = vector_weight,
        filter_weight = filter_weight,
        rerank_factor = rerank_factor,
        excluded = excluded,
    )
end

function search_database_locked(
    db::VectorDB,
    query::AbstractVector{<:Real};
    k::Int = 10,
    nprobe::Union{Nothing,Int} = nothing,
    filter::Union{Nothing,FilterExpr} = nothing,
    strategy::Symbol = :auto,
    planner_config::PlannerConfig = PlannerConfig(),
    postfilter_oversample::Int = 10,
    adaptive_postfilter::Bool = true,
    adaptive::Bool = true,
    max_nprobe::Union{Nothing,Int} = nothing,
    candidate_multiplier::Float64 = planner_config.candidate_multiplier,
    postfilter_candidate_multiplier::Float64 = planner_config.postfilter_candidate_multiplier,
    vector_weight::Float64 = 0.5,
    filter_weight::Float64 = 0.5,
    rerank_factor::Int = 4,
    _plan::Union{Nothing,QueryPlan} = nothing,
)
    length(query)==db.dim||throw(DimensionMismatch("query dimension doesnt match database"))
    k>0||throw(ArgumentError("k must be positive"))
    strategy===:auto||strategy_from_symbol(filter, strategy)
    exclusions=length(db.immutable_segments)>1 ? cached_database_segment_exclusions(db) :
               nothing
    index=db.index
    merged=SearchResult[]
    offset=0
    plan=nothing

    if index===nothing
        strategy in (:auto, :exact)||throw(
            ArgumentError("search strategy $(strategy) requires a built index"),
        )
    else
        plan=_plan===nothing ?
             (
            strategy===:auto ?
            cached_plan_query(db, filter; k = k, config = planner_config) :
            plan_query(db, filter; k = k, strategy = strategy, config = planner_config)
        ) : _plan
        resolved_minimum_nprobe=nprobe===nothing ? max(1, plan.minimum_nprobe) : nprobe
        resolved_nprobe=nprobe===nothing ? max(1, plan.nprobe) : nprobe
        list_count=length(index.ivf.lists)
        1<=resolved_nprobe<=list_count||throw(
            ArgumentError("nprobe must be between 1 and list count"),
        )
        resolved_max_nprobe=max_nprobe===nothing ? resolved_nprobe : max_nprobe
        resolved_max_nprobe=clamp(resolved_max_nprobe, resolved_nprobe, list_count)
        base_vectors=stored_vectors(db.vector_store)
        base_metadata=stored_metadata(db.metadata_store)
        primary_excluded=exclusions===nothing ? db.base_tombstones : first(exclusions)

        raw_results=if plan.strategy isa ExactStrategy||plan.strategy isa
                                                        PreFilterExactStrategy
            search_exact(
                base_vectors,
                base_metadata,
                query;
                k = k,
                metric = db.metric,
                filter = filter,
                filter_index = db.filter_index,
                vector_norms = index.ivf.vector_norms,
                excluded = primary_excluded,
            )
        elseif plan.strategy isa IVFStrategy
            search_ivf(
                index.ivf,
                base_vectors,
                base_metadata,
                query;
                k = k,
                nprobe = resolved_nprobe,
                metric = db.metric,
                excluded = primary_excluded,
            )
        elseif plan.strategy isa IVFPreFilterStrategy
            search_ivf_prefilter(
                index,
                base_vectors,
                base_metadata,
                query;
                k = k,
                nprobe = resolved_nprobe,
                metric = db.metric,
                filter = filter,
                excluded = primary_excluded,
            )
        elseif plan.strategy isa IVFPostFilterStrategy
            resolved_oversample=adaptive_postfilter ?
                                resolve_postfilter_oversample(
                postfilter_oversample,
                plan.selectivity;
                candidate_multiplier = postfilter_candidate_multiplier,
            ) : postfilter_oversample
            search_ivf_postfilter(
                index.ivf,
                base_vectors,
                base_metadata,
                query;
                k = k,
                nprobe = resolved_nprobe,
                metric = db.metric,
                filter = filter,
                oversample = resolved_oversample,
                excluded = primary_excluded,
            )
        elseif plan.strategy isa BoundedFilterAwareIVFStrategy
            search_filter_aware_bound(
                index,
                base_vectors,
                base_metadata,
                query;
                k = k,
                minimum_nprobe = resolved_minimum_nprobe,
                max_nprobe = resolved_max_nprobe,
                metric = db.metric,
                filter = filter,
                excluded = primary_excluded,
            )
        else
            minimum_nprobe=adaptive ? resolved_minimum_nprobe : resolved_nprobe
            search_filter_aware_ivf(
                index,
                base_vectors,
                base_metadata,
                query;
                k = k,
                nprobe = minimum_nprobe,
                metric = db.metric,
                filter = filter,
                adaptive = adaptive,
                max_nprobe = resolved_max_nprobe,
                candidate_multiplier = candidate_multiplier,
                vector_weight = vector_weight,
                filter_weight = filter_weight,
                rerank_factor = rerank_factor,
                excluded = primary_excluded,
            )
        end

        merged=database_search_results(raw_results, db.id_store)
        offset=length(db.vector_store)
    end

    first_extra=index===nothing ? 1 : 2

    for segment_index = first_extra:length(db.immutable_segments)
        segment=db.immutable_segments[segment_index]
        segment_excluded=if exclusions===nothing
            segment_index==1||error("segment exclusion table is incomplete")
            db.base_tombstones
        else
            segment_index<=length(exclusions)||error(
                "segment exclusion table is incomplete",
            )
            exclusions[segment_index]
        end
        raw=search_immutable_segment_raw(
            segment,
            query,
            plan;
            k = k,
            nprobe = nprobe,
            filter = filter,
            postfilter_oversample = postfilter_oversample,
            adaptive_postfilter = adaptive_postfilter,
            adaptive = adaptive,
            max_nprobe = max_nprobe,
            candidate_multiplier = candidate_multiplier,
            postfilter_candidate_multiplier = postfilter_candidate_multiplier,
            vector_weight = vector_weight,
            filter_weight = filter_weight,
            rerank_factor = rerank_factor,
            metric = db.metric,
            excluded = segment_excluded,
        )
        results=database_search_results(raw, segment.id_store; index_offset = offset)
        merged=merge_database_search_results(merged, results, k)
        offset+=length(segment.vector_store)
    end

    delta_vectors=stored_vectors(db.delta_store.vector_store)
    delta_metadata=stored_metadata(db.delta_store.metadata_store)
    length(delta_metadata)<=db.maintenance_config.max_delta_search_records||error(
        "delta search work exceeds the configured bound",
    )
    delta_raw=search_exact(
        delta_vectors,
        delta_metadata,
        query;
        k = k,
        metric = db.metric,
        filter = filter,
    )
    delta_results=database_search_results(
        delta_raw,
        db.delta_store.id_store;
        index_offset = offset,
    )
    return merge_database_search_results(merged, delta_results, k)
end

function search(
    db::VectorDB,
    query::AbstractVector{<:Real};
    filter::Union{Nothing,NamedTuple,FilterExpr} = nothing,
    kwargs...,
)
    normalized_filter=normalize_filter(filter)

    return with_database_read(db.database_lock) do
        validate_database_fast(db)
        converted=convert_validated_vector(query, db.dim, db.metric; context = "query")
        search_database_locked(db, converted; filter = normalized_filter, kwargs...)
    end
end

const DATABASE_BATCH_WORKER_KEY=:sen_database_batch_worker
const DEFAULT_BATCH_PARALLEL_THRESHOLD=4

function database_batch_worker_owned()
    return get(task_local_storage(), DATABASE_BATCH_WORKER_KEY, false)===true
end

function with_database_batch_worker(f::Function)
    storage=task_local_storage()
    had_owner=haskey(storage, DATABASE_BATCH_WORKER_KEY)
    previous_owner=get(storage, DATABASE_BATCH_WORKER_KEY, false)
    storage[DATABASE_BATCH_WORKER_KEY]=true

    try
        return f()
    finally
        if had_owner
            storage[DATABASE_BATCH_WORKER_KEY]=previous_owner
        else
            delete!(storage, DATABASE_BATCH_WORKER_KEY)
        end
    end
end

function resolve_database_batch_workers(
    query_count::Int;
    parallel::Bool,
    workers::Int,
    parallel_threshold::Int,
)
    workers>0||throw(ArgumentError("batch workers must be positive"))
    parallel_threshold>0||throw(ArgumentError("batch parallel threshold must be positive"))

    if !parallel||query_count<parallel_threshold||database_batch_worker_owned()
        return 1
    end

    return max(1, min(query_count, workers, Threads.nthreads(:default)))
end

function resolve_database_batch_plan(
    db::VectorDB,
    filter::Union{Nothing,FilterExpr},
    kwargs,
)
    db.index===nothing&&return nothing
    k=get(kwargs, :k, 10)
    strategy=get(kwargs, :strategy, :auto)
    planner_config=get(kwargs, :planner_config, PlannerConfig())
    return strategy===:auto ?
           cached_plan_query(db, filter; k = k, config = planner_config) :
           plan_query(db, filter; k = k, strategy = strategy, config = planner_config)
end

function search_database_exact_batch_locked(db::VectorDB, queries::AbstractMatrix; k::Int)
    base_vectors=stored_vectors(db.vector_store)
    base_metadata=stored_metadata(db.metadata_store)
    vector_norms=db.index===nothing ? nothing : db.index.ivf.vector_norms
    base_raw=search_exact_batch(
        base_vectors,
        base_metadata,
        queries;
        k = k,
        metric = db.metric,
        vector_norms = vector_norms,
        excluded = db.index===nothing ? nothing : db.base_tombstones,
    )
    delta_vectors=stored_vectors(db.delta_store.vector_store)
    delta_metadata=stored_metadata(db.delta_store.metadata_store)
    length(delta_metadata)<=db.maintenance_config.max_delta_search_records||error(
        "delta search work exceeds the configured bound",
    )
    delta_raw=search_exact_batch(
        delta_vectors,
        delta_metadata,
        queries;
        k = k,
        metric = db.metric,
    )
    results=Vector{Vector{SearchResult}}(undef, size(queries, 2))

    for query_index in axes(queries, 2)
        base_results=database_search_results(base_raw[query_index], db.id_store)
        delta_results=database_search_results(
            delta_raw[query_index],
            db.delta_store.id_store;
            index_offset = length(db.vector_store),
        )
        results[query_index]=merge_database_search_results(base_results, delta_results, k)
    end

    return results
end

function use_exact_batch_search(
    db::VectorDB,
    filter::Union{Nothing,FilterExpr},
    plan,
    kwargs,
)
    filter===nothing||return false
    length(db.immutable_segments)<=1||return false
    haskey(kwargs, :nprobe)&&return false
    haskey(kwargs, :max_nprobe)&&return false
    strategy=get(kwargs, :strategy, :auto)
    db.index===nothing&&return strategy in (:auto, :exact)
    return plan!==nothing&&plan.strategy isa ExactStrategy
end

function search_database_batch_parallel!(
    results::Vector{Vector{SearchResult}},
    db::VectorDB,
    queries::AbstractMatrix{<:Real},
    worker_count::Int;
    kwargs...,
)
    query_count=size(queries, 2)
    jobs=Channel{Int}(query_count)

    for index = 1:query_count
        put!(jobs, index)
    end

    close(jobs)
    failures=Vector{Union{Nothing,Tuple{Int,Exception}}}(nothing, worker_count)
    tasks=Vector{Task}(undef, worker_count)

    for worker_index = 1:worker_count
        tasks[worker_index]=Threads.@spawn with_database_batch_worker() do
            query_index=0

            try
                for index in jobs
                    query_index=index
                    results[index]=search_database_locked(
                        db,
                        @view(queries[:, index]);
                        kwargs...,
                    )
                end
            catch error
                failures[worker_index]=(query_index, error)
            end
        end
    end

    wait.(tasks)
    failure=nothing

    for current in failures
        current===nothing&&continue

        if failure===nothing||current[1]<failure[1]
            failure=current
        end
    end

    failure===nothing||throw(failure[2])
    return results
end

function search(
    db::VectorDB,
    queries::AbstractMatrix{<:Real};
    filter::Union{Nothing,NamedTuple,FilterExpr} = nothing,
    parallel::Bool = true,
    workers::Int = Threads.nthreads(:default),
    parallel_threshold::Int = DEFAULT_BATCH_PARALLEL_THRESHOLD,
    kwargs...,
)
    normalized_filter=normalize_filter(filter)

    return with_database_read(db.database_lock) do
        validate_database_fast(db)
        converted=convert_validated_vectors(queries, db.dim, db.metric; context = "query")
        query_count=size(converted, 2)
        results=Vector{Vector{SearchResult}}(undef, query_count)
        worker_count=resolve_database_batch_workers(
            query_count;
            parallel = parallel,
            workers = workers,
            parallel_threshold = parallel_threshold,
        )
        query_count==0&&return results
        batch_plan=resolve_database_batch_plan(db, normalized_filter, kwargs)

        if use_exact_batch_search(db, normalized_filter, batch_plan, kwargs)
            return search_database_exact_batch_locked(
                db,
                converted;
                k = get(kwargs, :k, 10),
            )
        end

        if worker_count==1
            for index = 1:query_count
                results[index]=search_database_locked(
                    db,
                    @view(converted[:, index]);
                    filter = normalized_filter,
                    _plan = batch_plan,
                    kwargs...,
                )
            end

            return results
        end

        return search_database_batch_parallel!(
            results,
            db,
            converted,
            worker_count;
            filter = normalized_filter,
            _plan = batch_plan,
            kwargs...,
        )
    end
end

function save!(db::VectorDB; retain_snapshots::Int = 2)
    return with_database_write(db.database_lock) do
        activate_database_writer_lock!(db)
        validate_database(db)
        retain_snapshots>0||throw(ArgumentError("retain snapshots must be positive"))
        config=db.build_config
        segmented=db.segment_mode
        current_index=has_usable_base(db)&&db.index_revision==db.revision
        stores=segmented ? nothing :
               (
            current_index ?
            (
                vector_store = db.vector_store,
                metadata_store = db.metadata_store,
                id_store = db.id_store,
            ) : materialize_database(db)
        )
        count=segmented ? logical_length(db) : length(stores.vector_store)
        persisted_nlists=config===nothing||count==0 ? nothing : min(config.nlists, count)
        persisted_training_count=persisted_nlists===nothing ? count :
                                 min(count, max(persisted_nlists, config.training_count))
        manifest=create_database_manifest(
            db.dim,
            db.metric,
            count;
            format_version = segmented ? 3 : 2,
            nlists = persisted_nlists,
            revision = db.revision,
            index_revision = segmented ? db.index_revision :
                             (current_index ? db.index_revision : nothing),
            iterations = config===nothing ? 20 : config.iterations,
            seed = config===nothing ? 42 : config.seed,
            restarts = config===nothing ? 1 : config.restarts,
            training_count = persisted_training_count,
        )
        snapshot=new_database_snapshot(db.path)

        try
            save_manifest(snapshot.temporary_path, manifest)

            if segmented
                save_database_segments(snapshot.temporary_path, db)
            else
                save_vector_store(snapshot.temporary_path, stores.vector_store)
                save_metadata_store(snapshot.temporary_path, stores.metadata_store)
                save_id_store(snapshot.temporary_path, stores.id_store)
                current_index&&save_ivf_index(snapshot.temporary_path, db.index.ivf)
            end

            seal_database_snapshot(snapshot.temporary_path, db.revision)
            maybe_pause_database_process!(:after_snapshot_seal, db.path)
            commit_database_snapshot(db.path, snapshot)
            maybe_pause_database_process!(:after_snapshot_commit, db.path)
            checkpoint_database_wal!(db)
        catch
            abort_database_snapshot(snapshot)
            rethrow()
        end

        prune_database_snapshots(db.path; retain = retain_snapshots)
        return db
    end
end

function install_loaded_database_segments!(
    db::VectorDB,
    topology;
    live_count::Int = db.live_count,
)
    old_vector_store=db.vector_store
    db.immutable_segments=topology.immutable_segments
    primary=first(db.immutable_segments)
    db.vector_store=primary.vector_store
    db.metadata_store=primary.metadata_store
    db.id_store=primary.id_store
    db.base_tombstones=copy(primary.excluded)
    db.index=primary.index
    db.filter_index=primary.index===nothing ? nothing : primary.filter_index
    db.build_config=primary.build_config
    db.index_revision=topology.index_revision
    db.index_bytes=primary.index===nothing ? 0 : primary.index_bytes
    db.active_segment=topology.active_segment
    db.delta_store=db.active_segment.store
    db.segment_mode=true
    db.live_count=live_count
    db.base_tombstones=first(database_segment_exclusions(db))
    old_vector_store===db.vector_store||release_vector_store_mapping!(old_vector_store)
    clear_plan_cache!(db)
    return db
end

function load_db_from_snapshot(
    path::AbstractString,
    snapshot_path::AbstractString;
    iterations::Int = 20,
    seed::Int = 42,
    rebuild::Bool = false,
    checkpoint_operations::Int = 10_000,
    checkpoint_bytes::Int = 64*1024*1024,
    checkpoint_retain_snapshots::Int = 2,
    maintenance_config::MaintenanceConfig = MaintenanceConfig(),
    mmap_vectors::Union{Bool,Symbol} = :auto,
    mmap_threshold_bytes::Int = DEFAULT_VECTOR_MMAP_THRESHOLD_BYTES,
)
    descriptor=validate_database_snapshot(snapshot_path)
    manifest=load_manifest(snapshot_path)
    descriptor.revision===nothing||descriptor.revision==manifest.revision||throw(
        ArgumentError("snapshot revision doesnt match manifest"),
    )
    topology=load_database_segments(
        snapshot_path,
        manifest.dim,
        manifest.metric,
        manifest.revision;
        mmap_vectors = mmap_vectors,
        mmap_threshold_bytes = mmap_threshold_bytes,
    )

    if manifest.format_version>=3
        topology===nothing&&throw(
            ArgumentError("segment-native snapshot is missing its segment topology"),
        )
        topology.index_revision==manifest.index_revision||throw(
            ArgumentError("segment index revision doesnt match manifest"),
        )
    end

    db=if topology===nothing
        vector_store=load_vector_store(
            snapshot_path;
            mmap = mmap_vectors,
            mmap_threshold_bytes = mmap_threshold_bytes,
        )
        metadata_store=load_metadata_store(snapshot_path)
        id_store=load_id_store(snapshot_path)

        vector_store.dim==manifest.dim||throw(
            DimensionMismatch("stored vector dimension doesnt match manifest"),
        )
        length(vector_store)==manifest.count||throw(
            DimensionMismatch("stored vector count doesnt match manifest"),
        )
        length(metadata_store)==manifest.count||throw(
            DimensionMismatch("stored metadata count doesnt match manifest"),
        )
        length(id_store)==manifest.count||throw(
            DimensionMismatch("stored id count doesnt match manifest"),
        )
        validate_stored_vectors!(stored_vectors(vector_store), manifest.metric)

        VectorDB(
            String(path),
            manifest.dim,
            manifest.metric,
            vector_store,
            metadata_store,
            id_store,
            nothing,
            nothing,
            manifest.build_config,
            manifest.revision,
            nothing,
            DatabaseLock(),
            Dict{Any,Any}(),
            ReentrantLock();
            checkpoint_operations = checkpoint_operations,
            checkpoint_bytes = checkpoint_bytes,
            checkpoint_retain_snapshots = checkpoint_retain_snapshots,
            maintenance_config = maintenance_config,
        )
    else
        length(
            segment_topology_visible_ids(
                topology.immutable_segments,
                topology.active_segment,
            ),
        )==manifest.count||throw(
            DimensionMismatch("stored segment topology count doesnt match manifest"),
        )
        primary=first(topology.immutable_segments)
        database=VectorDB(
            String(path),
            manifest.dim,
            manifest.metric,
            primary.vector_store,
            primary.metadata_store,
            primary.id_store,
            primary.index,
            primary.index===nothing ? nothing : primary.filter_index,
            primary.build_config,
            manifest.revision,
            topology.index_revision,
            DatabaseLock(),
            Dict{Any,Any}(),
            ReentrantLock();
            checkpoint_operations = checkpoint_operations,
            checkpoint_bytes = checkpoint_bytes,
            checkpoint_retain_snapshots = checkpoint_retain_snapshots,
            maintenance_config = maintenance_config,
        )
        install_loaded_database_segments!(database, topology; live_count = manifest.count)
    end

    if topology===nothing&&manifest.index_revision==manifest.revision&&isfile(
        index_file_path(snapshot_path),
    )
        ivf=load_ivf_index(snapshot_path)
        size(ivf.centroids, 1)==manifest.dim||throw(
            DimensionMismatch("stored index dimension doesnt match manifest"),
        )
        manifest.nlists===nothing||length(ivf.lists)==manifest.nlists||throw(
            DimensionMismatch("stored list count doesnt match manifest"),
        )
        sum(length, ivf.lists)==manifest.count||throw(
            DimensionMismatch("stored index count doesnt match manifest"),
        )
        ivf.metric===manifest.metric||throw(
            ArgumentError("stored index metric doesnt match manifest"),
        )
        db.index=build_filter_aware_ivf(ivf, stored_metadata(db.metadata_store))
        db.filter_index=build_bitset_index(stored_metadata(db.metadata_store))
        db.index_revision=manifest.index_revision
        db.index_bytes=database_index_bytes(db.index, db.filter_index)
    end

    replay_database_wal!(db)

    if rebuild&&!is_built(db)&&db.build_config!==nothing&&length(db)>0
        config=db.build_config
        nlists=min(config.nlists, length(db))
        build!(
            db;
            nlists = nlists,
            iterations = config.iterations,
            seed = config.seed,
            restarts = config.restarts,
            training_count = min(length(db), max(nlists, config.training_count)),
        )
    end

    ensure_delta_search_capacity!(db, Any[])
    db.segment_mode&&active_segment_work(db.active_segment)>=db.maintenance_config.active_segment_threshold&&seal_active_segment_locked!(
        db,
    )
    validate_database(db)
    return db
end

function load_db_from_wal(
    path::AbstractString;
    checkpoint_operations::Int = 10_000,
    checkpoint_bytes::Int = 64*1024*1024,
    checkpoint_retain_snapshots::Int = 2,
    maintenance_config::MaintenanceConfig = MaintenanceConfig(),
)
    wal=read_database_wal(path; repair_tail = true)
    wal===nothing&&throw(ArgumentError("database WAL does not exist"))
    wal.header.revision==UInt64(0)||throw(
        ArgumentError("database snapshot is missing for checkpointed WAL"),
    )
    db=VectorDB(
        String(path),
        wal.header.dim,
        wal.header.metric,
        create_vector_store(wal.header.dim),
        create_metadata_store(),
        create_id_store(),
        nothing,
        nothing,
        nothing,
        UInt64(0),
        nothing,
        DatabaseLock(),
        Dict{Any,Any}(),
        ReentrantLock();
        checkpoint_operations = checkpoint_operations,
        checkpoint_bytes = checkpoint_bytes,
        checkpoint_retain_snapshots = checkpoint_retain_snapshots,
        maintenance_config = maintenance_config,
    )
    replay_database_wal!(db)
    ensure_delta_search_capacity!(db, Any[])
    db.segment_mode&&active_segment_work(db.active_segment)>=db.maintenance_config.active_segment_threshold&&seal_active_segment_locked!(
        db,
    )
    validate_database(db)
    return db
end

function rebase_recovered_database_wal!(db::VectorDB)
    checkpoint_revision=db.wal_checkpoint_revision
    checkpoint_revision===nothing&&return db
    checkpoint_revision<=db.revision&&return db
    replace_database_wal(db.path, db.revision, db.dim, db.metric)
    db.wal_revision=db.revision
    db.wal_checkpoint_revision=db.revision
    return db
end

function load_db_without_writer_lock(
    path::AbstractString;
    iterations::Int = 20,
    seed::Int = 42,
    rebuild::Bool = false,
    recover::Bool = false,
    checkpoint_operations::Int = 10_000,
    checkpoint_bytes::Int = 64*1024*1024,
    checkpoint_retain_snapshots::Int = 2,
    maintenance_config::MaintenanceConfig = MaintenanceConfig(),
    mmap_vectors::Union{Bool,Symbol} = :auto,
    mmap_threshold_bytes::Int = DEFAULT_VECTOR_MMAP_THRESHOLD_BYTES,
)
    if isfile(database_wal_path(path))&&!isfile(database_current_path(path))&&!isfile(
        joinpath(path, "manifest.toml"),
    )&&isempty(database_snapshot_generations(path))
        return load_db_from_wal(
            path;
            checkpoint_operations = checkpoint_operations,
            checkpoint_bytes = checkpoint_bytes,
            checkpoint_retain_snapshots = checkpoint_retain_snapshots,
            maintenance_config = maintenance_config,
        )
    end

    recovered_snapshot=false
    snapshot_path=try
        current_database_snapshot(path)
    catch
        recover||rethrow()
        recovered_snapshot=true
        recover_database_snapshot(path)
    end

    try
        db=load_db_from_snapshot(
            path,
            snapshot_path;
            iterations = iterations,
            seed = seed,
            rebuild = rebuild,
            checkpoint_operations = checkpoint_operations,
            checkpoint_bytes = checkpoint_bytes,
            checkpoint_retain_snapshots = checkpoint_retain_snapshots,
            maintenance_config = maintenance_config,
            mmap_vectors = mmap_vectors,
            mmap_threshold_bytes = mmap_threshold_bytes,
        )
        recovered_snapshot&&rebase_recovered_database_wal!(db)
        return db
    catch
        recover||rethrow()
        recovered_path=recover_database_snapshot(path)
        db=load_db_from_snapshot(
            path,
            recovered_path;
            iterations = iterations,
            seed = seed,
            rebuild = rebuild,
            checkpoint_operations = checkpoint_operations,
            checkpoint_bytes = checkpoint_bytes,
            checkpoint_retain_snapshots = checkpoint_retain_snapshots,
            maintenance_config = maintenance_config,
            mmap_vectors = mmap_vectors,
            mmap_threshold_bytes = mmap_threshold_bytes,
        )
        rebase_recovered_database_wal!(db)
        return db
    end
end

function load_db(
    path::AbstractString;
    iterations::Int = 20,
    seed::Int = 42,
    rebuild::Bool = false,
    recover::Bool = false,
    checkpoint_operations::Int = 10_000,
    checkpoint_bytes::Int = 64*1024*1024,
    checkpoint_retain_snapshots::Int = 2,
    maintenance_config::MaintenanceConfig = MaintenanceConfig(),
    mmap_vectors::Union{Bool,Symbol} = :auto,
    mmap_threshold_bytes::Int = DEFAULT_VECTOR_MMAP_THRESHOLD_BYTES,
)
    io=acquire_database_writer_lock(path)

    try
        db=load_db_without_writer_lock(
            path;
            iterations = iterations,
            seed = seed,
            rebuild = rebuild,
            recover = recover,
            checkpoint_operations = checkpoint_operations,
            checkpoint_bytes = checkpoint_bytes,
            checkpoint_retain_snapshots = checkpoint_retain_snapshots,
            maintenance_config = maintenance_config,
            mmap_vectors = mmap_vectors,
            mmap_threshold_bytes = mmap_threshold_bytes,
        )
        attach_database_writer_lock!(db, io)
        maybe_schedule_database_maintenance!(db)
        maybe_schedule_segment_indexing!(db)
        return db
    catch
        release_database_writer_lock(io)
        rethrow()
    end
end

function recover_db(
    path::AbstractString;
    iterations::Int = 20,
    seed::Int = 42,
    rebuild::Bool = false,
    checkpoint_operations::Int = 10_000,
    checkpoint_bytes::Int = 64*1024*1024,
    checkpoint_retain_snapshots::Int = 2,
    maintenance_config::MaintenanceConfig = MaintenanceConfig(),
    mmap_vectors::Union{Bool,Symbol} = :auto,
    mmap_threshold_bytes::Int = DEFAULT_VECTOR_MMAP_THRESHOLD_BYTES,
)
    return load_db(
        path;
        iterations = iterations,
        seed = seed,
        rebuild = rebuild,
        recover = true,
        checkpoint_operations = checkpoint_operations,
        checkpoint_bytes = checkpoint_bytes,
        checkpoint_retain_snapshots = checkpoint_retain_snapshots,
        maintenance_config = maintenance_config,
        mmap_vectors = mmap_vectors,
        mmap_threshold_bytes = mmap_threshold_bytes,
    )
end
