function create_db(path::AbstractString;dim::Int,metric::Symbol=:cosine,initial_capacity::Int=0,durable::Bool=true,checkpoint_operations::Int=10_000,checkpoint_bytes::Int=64*1024*1024,checkpoint_retain_snapshots::Int=2,maintenance_config::MaintenanceConfig=MaintenanceConfig(),)
    dim>0||throw(ArgumentError("dimension must be positive"))
    metric in (:cosine,:dot)||throw(ArgumentError("metric must be :cosine or :dot"))
    initial_capacity>=0||throw(ArgumentError("initial capacity cannot be negative"))

    vector_store=create_vector_store(dim;initial_capacity=initial_capacity,)
    metadata_store=create_metadata_store(initial_capacity=initial_capacity,)
    id_store=create_id_store(initial_capacity=initial_capacity,)

    db=VectorDB(String(path),dim,metric,vector_store,metadata_store,id_store,nothing,nothing,nothing,UInt64(0),nothing,DatabaseLock(),Dict{Any,Any}(),ReentrantLock();checkpoint_operations=checkpoint_operations,checkpoint_bytes=checkpoint_bytes,checkpoint_retain_snapshots=checkpoint_retain_snapshots,maintenance_config=maintenance_config,)

    if durable
        io=acquire_database_writer_lock(path)

        try
            wal_path=database_wal_path(path)
            current_path=database_current_path(path)
            manifest_path=joinpath(path,"manifest.toml")
            any(isfile,(wal_path,current_path,manifest_path))&&throw(ArgumentError("database already exists"))
            replace_database_wal(path,db.revision,db.dim,db.metric)
            db.wal_revision=db.revision
            db.wal_checkpoint_revision=db.revision
            attach_database_writer_lock!(db,io)
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
    count==length(db.metadata_store)==length(db.id_store)||error("database stores are misaligned")
    length(db.id_store.positions)==count||error("database id positions are misaligned")
    length(db.base_tombstones)==count||error("database tombstones are misaligned")
    validate_delta_store(db.delta_store)
    0<=db.live_count<=count+length(db.delta_store)||error("database live count is invalid")

    if db.index===nothing
        db.filter_index===nothing||error("database filter index exists without vector index")
        db.index_revision===nothing||error("database index revision exists without vector index")
        db.index_bytes==0||error("database index size exists without vector index")
        isempty(db.delta_store.id_store.ids)||error("unindexed database has delta records")
        db.live_count==count||error("unindexed database live count is misaligned")
    else
        db.filter_index===nothing&&error("database vector index exists without filter index")
        db.index_revision===nothing&&error("database vector index has no revision")
        db.index_bytes>0||error("database vector index size is invalid")
        db.index_revision<=db.revision||error("database index revision is ahead of database")
        sum(length,db.index.ivf.lists)==count||error("database index count is misaligned")
        db.filter_index.count==count||error("database filter index count is misaligned")
    end

    return db
end

function validate_database(db::VectorDB)
    validate_database_fast(db)

    if !isempty(db.mutation_history)
        last(db.mutation_history).revision==db.revision||error("mutation history does not reach the current database revision")

        for index in 2:length(db.mutation_history)
            db.mutation_history[index].revision==db.mutation_history[index-1].revision+UInt64(1)||error("mutation history revisions are not contiguous")
        end

        db.index_revision===nothing||first(db.mutation_history).revision==db.index_revision+UInt64(1)||error("mutation history does not follow the index revision")
    end

    for(position,id) in enumerate(db.id_store.ids)
        get(db.id_store.positions,id,0)==position||error("database id positions are misaligned")
    end

    for(position,id) in enumerate(db.delta_store.id_store.ids)
        get(db.delta_store.id_store.positions,id,0)==position||error("delta id positions are misaligned")
        has_id(db.id_store,id)&&!db.base_tombstones[get_position(db.id_store,id)]&&error("database id exists in base and delta")
    end

    db.live_count==length(db.vector_store)-count(db.base_tombstones)+length(db.delta_store)||error("database live count is misaligned")

    return db
end

function next_database_revision(db::VectorDB)
    db.revision==typemax(UInt64)&&error("database revision overflow")
    return db.revision+UInt64(1)
end

const DATABASE_MUTATION_FAULT_KEY=:sen_database_mutation_fault
const DATABASE_MUTATION_FAULT_STAGES=(:before_wal,:after_wal,:during_apply,:after_apply,)

struct InjectedDatabaseMutationError <: Exception
    stage::Symbol
end

struct DatabaseMutationCommittedError <: Exception
    revision::UInt64
    cause::Exception
end

function Base.showerror(io::IO,error::InjectedDatabaseMutationError)
    print(io,"injected database mutation failure at ",error.stage)
end

function Base.showerror(io::IO,error::DatabaseMutationCommittedError)
    print(io,"database mutation revision ",error.revision," committed durably but its in-memory application failed: ")
    showerror(io,error.cause)
end

function inject_database_mutation_fault!(stage::Symbol)
    stage in DATABASE_MUTATION_FAULT_STAGES||throw(ArgumentError("unsupported database mutation fault stage"))
    task_local_storage()[DATABASE_MUTATION_FAULT_KEY]=stage
    return stage
end

function clear_database_mutation_fault!()
    storage=task_local_storage()
    haskey(storage,DATABASE_MUTATION_FAULT_KEY)&&delete!(storage,DATABASE_MUTATION_FAULT_KEY)
    return nothing
end

function database_mutation_fault_stage()
    return get(task_local_storage(),DATABASE_MUTATION_FAULT_KEY,nothing)
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

function logical_length(db::VectorDB)
    return db.live_count
end

function delta_search_work(db::VectorDB)
    return length(db.delta_store)
end

function ensure_delta_search_capacity!(db::VectorDB,ids::AbstractVector)
    has_usable_base(db)||return db
    limit=db.maintenance_config.max_delta_search_records
    additions=count(id->!has_delta_id(db.delta_store,id),ids)
    delta_search_work(db)+additions<=limit&&return db
    length(ids)<=limit||throw(ArgumentError("mutation would exceed the configured delta search bound"))
    rebuild_database_maintenance!(db)
    isempty(db.delta_store.id_store.ids)||error("foreground delta compaction did not clear the mutation tail")
    length(ids)<=limit||error("mutation exceeds the configured delta search bound after compaction")
    return db
end

function has_database_id(db::VectorDB,id)
    has_delta_id(db.delta_store,id)&&return true
    has_id(db.id_store,id)||return false
    return !db.base_tombstones[get_position(db.id_store,id)]
end

function next_database_id(db::VectorDB)
    id=logical_length(db)+1

    while has_database_id(db,id)
        id+=1
    end

    return id
end

function tombstone_base_id!(db::VectorDB,id)
    position=get_position(db.id_store,id)
    db.base_tombstones[position]=true
    return position
end

function finish_database_mutation!(db::VectorDB,revision::UInt64,wal_body::Vector{UInt8})
    revision==next_database_revision(db)||error("database mutation revision is invalid")
    db.revision=revision
    record_database_mutation!(db,revision,wal_body)
    clear_plan_cache!(db)
    validate_database_fast(db)
    maybe_checkpoint_database!(db)
    maybe_schedule_database_maintenance!(db)
    return db
end

function database_mutation_state(db::VectorDB)
    return(
        vector_store=deepcopy(db.vector_store),
        metadata_store=deepcopy(db.metadata_store),
        id_store=deepcopy(db.id_store),
        index=deepcopy(db.index),
        filter_index=deepcopy(db.filter_index),
        build_config=deepcopy(db.build_config),
        revision=db.revision,
        index_revision=db.index_revision,
        index_bytes=db.index_bytes,
        delta_store=deepcopy(db.delta_store),
        base_tombstones=copy(db.base_tombstones),
        live_count=db.live_count,
        mutation_history=deepcopy(db.mutation_history),
        wal_revision=db.wal_revision,
        wal_checkpoint_revision=db.wal_checkpoint_revision,
    )
end

function install_database_mutation_state!(db::VectorDB,state)
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
    db.live_count=state.live_count
    db.mutation_history=state.mutation_history
    db.wal_revision=state.wal_revision
    db.wal_checkpoint_revision=state.wal_checkpoint_revision
    old_vector_store===db.vector_store||release_vector_store_mapping!(old_vector_store)
    clear_plan_cache!(db)
    validate_database(db)
    return db
end

function recover_committed_database_mutation!(db::VectorDB)
    recovered=load_db_without_writer_lock(
        db.path;
        rebuild=false,
        recover=true,
        checkpoint_operations=db.checkpoint_operations,
        checkpoint_bytes=db.checkpoint_bytes,
        checkpoint_retain_snapshots=db.checkpoint_retain_snapshots,
        maintenance_config=db.maintenance_config,
        mmap_vectors=false,
    )
    return install_database_mutation_state!(db,database_mutation_state(recovered))
end

function commit_database_mutation!(apply::Function,db::VectorDB,revision::UInt64,wal_body::Vector{UInt8})
    maybe_inject_database_mutation_fault!(:before_wal)
    durable=append_database_wal_body_if_active!(db,revision,wal_body)
    fault_stage=database_mutation_fault_stage()
    backup=!durable&&fault_stage in (:during_apply,:after_apply,) ? database_mutation_state(db) : nothing

    try
        maybe_inject_database_mutation_fault!(:after_wal)
        result=apply()
        maybe_inject_database_mutation_fault!(:after_apply)
        finish_database_mutation!(db,revision,wal_body)
        return result
    catch error
        if durable
            recover_committed_database_mutation!(db)
            throw(DatabaseMutationCommittedError(revision,error))
        elseif backup!==nothing
            install_database_mutation_state!(db,backup)
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

function Base.insert!(db::VectorDB,vector::AbstractVector{<:Real},metadata::NamedTuple;id=nothing,)
    return with_database_write(db.database_lock) do
        validate_database_fast(db)
        converted=convert_validated_vector(vector,db.dim,db.metric)
        resolved_id=id===nothing ? next_database_id(db) : id
        resolved_id===nothing&&throw(ArgumentError("id cannot be nothing"))
        has_database_id(db,resolved_id)&&throw(ArgumentError("id already exists"))
        ensure_delta_search_capacity!(db,[resolved_id])
        revision=next_database_revision(db)
        wal_body=database_wal_put_body(revision,[converted],[metadata],[resolved_id])
        has_usable_base(db)||make_vector_store_writable!(db.vector_store)

        return commit_database_mutation!(db,revision,wal_body) do
            insert_database_record!(db,converted,metadata,resolved_id)
            maybe_inject_database_mutation_fault!(:during_apply)
            db.live_count+=1
            return resolved_id
        end
    end
end

function prepare_database_batch(db::VectorDB,vectors::AbstractMatrix,metadata::AbstractVector{<:NamedTuple},ids;allow_existing::Bool,)
    _,count=size(vectors)
    length(metadata)==count||throw(DimensionMismatch("metadata count doesnt match vector count"))
    converted=convert_validated_vectors(vectors,db.dim,db.metric)
    converted_metadata=NamedTuple[value for value in metadata]
    resolved_ids=if ids===nothing
        generated=Vector{Any}(undef,count)
        reserved=Set{Any}()
        candidate=next_database_id(db)

        for index in 1:count
            while has_database_id(db,candidate)||candidate in reserved
                candidate+=1
            end

            generated[index]=candidate
            push!(reserved,candidate)
            candidate+=1
        end

        generated
    else
        collect(ids)
    end

    length(resolved_ids)==count||throw(DimensionMismatch("id count doesnt match vector count"))
    all(id->id!==nothing,resolved_ids)||throw(ArgumentError("id cannot be nothing"))
    length(Set{Any}(resolved_ids))==count||throw(ArgumentError("batch ids must be unique"))

    if !allow_existing
        any(id->has_database_id(db,id),resolved_ids)&&throw(ArgumentError("id already exists"))
    end

    return(vectors=converted,metadata=converted_metadata,ids=resolved_ids,count=count,)
end

function insert_database_record!(db::VectorDB,vector::AbstractVector{<:Real},metadata::NamedTuple,id)
    if has_usable_base(db)
        insert_delta!(db.delta_store,vector,metadata,id)
    else
        make_vector_store_writable!(db.vector_store)
        vector_position=insert_vector!(db.vector_store,vector)
        metadata_position=insert_metadata!(db.metadata_store,metadata)
        id_position=insert_id!(db.id_store,id)
        vector_position==metadata_position==id_position||error("database stores are misaligned")
        push!(db.base_tombstones,false)
    end

    return id
end

function update_database_record!(db::VectorDB,id,vector::AbstractVector{<:Real},metadata::NamedTuple)
    if has_delta_id(db.delta_store,id)
        update_delta!(db.delta_store,id;vector=vector,metadata=metadata,)
    elseif has_id(db.id_store,id)&&!db.base_tombstones[get_position(db.id_store,id)]
        position=get_position(db.id_store,id)

        if has_usable_base(db)
            tombstone_base_id!(db,id)
            insert_delta!(db.delta_store,vector,metadata,id)
        else
            make_vector_store_writable!(db.vector_store)
            update_vector!(db.vector_store,position,vector)
            update_metadata!(db.metadata_store,position,metadata)
        end
    else
        insert_database_record!(db,vector,metadata,id)
    end

    return id
end

function swap_delete_tombstone!(tombstones::BitVector,position::Int)
    last_position=length(tombstones)
    position==last_position||(tombstones[position]=tombstones[last_position])
    pop!(tombstones)
    return tombstones
end

function delete_database_record!(db::VectorDB,id)
    if has_delta_id(db.delta_store,id)
        delete_delta!(db.delta_store,id)
        return db
    end

    has_id(db.id_store,id)||throw(KeyError(id))
    position=get_position(db.id_store,id)
    db.base_tombstones[position]&&throw(KeyError(id))

    if has_usable_base(db)
        db.base_tombstones[position]=true
    else
        make_vector_store_writable!(db.vector_store)
        swap_delete_vector!(db.vector_store,position)
        swap_delete_metadata!(db.metadata_store,position)
        deleted=swap_delete_id!(db.id_store,id)
        deleted.position==position||error("database stores are misaligned")
        swap_delete_tombstone!(db.base_tombstones,position)
    end

    return db
end

function Base.insert!(db::VectorDB,vectors::AbstractMatrix,metadata::AbstractVector{<:NamedTuple};ids=nothing,)
    return with_database_write(db.database_lock) do
        validate_database_fast(db)
        batch=prepare_database_batch(db,vectors,metadata,ids;allow_existing=false,)
        batch.count==0&&return batch.ids
        ensure_delta_search_capacity!(db,batch.ids)
        revision=next_database_revision(db)
        wal_vectors=[@view batch.vectors[:,index] for index in 1:batch.count]
        wal_body=database_wal_put_body(revision,wal_vectors,batch.metadata,batch.ids)
        has_usable_base(db)||make_vector_store_writable!(db.vector_store)

        return commit_database_mutation!(db,revision,wal_body) do
            for index in 1:batch.count
                insert_database_record!(db,@view(batch.vectors[:,index]),batch.metadata[index],batch.ids[index])
                maybe_inject_database_mutation_fault!(:during_apply)
            end

            db.live_count+=batch.count
            return batch.ids
        end
    end
end

function upsert!(db::VectorDB,vector::AbstractVector{<:Real},metadata::NamedTuple;id,)
    return with_database_write(db.database_lock) do
        validate_database_fast(db)
        id===nothing&&throw(ArgumentError("id cannot be nothing"))
        converted=convert_validated_vector(vector,db.dim,db.metric)
        existing=has_database_id(db,id)
        ensure_delta_search_capacity!(db,[id])
        revision=next_database_revision(db)
        wal_body=database_wal_put_body(revision,[converted],[metadata],[id])
        has_usable_base(db)||make_vector_store_writable!(db.vector_store)

        return commit_database_mutation!(db,revision,wal_body) do
            update_database_record!(db,id,converted,metadata)
            maybe_inject_database_mutation_fault!(:during_apply)
            existing||(db.live_count+=1)
            return id
        end
    end
end

function upsert!(db::VectorDB,vectors::AbstractMatrix,metadata::AbstractVector{<:NamedTuple};ids,)
    return with_database_write(db.database_lock) do
        validate_database_fast(db)
        batch=prepare_database_batch(db,vectors,metadata,ids;allow_existing=true,)
        batch.count==0&&return batch.ids
        new_count=count(id->!has_database_id(db,id),batch.ids)
        ensure_delta_search_capacity!(db,batch.ids)
        revision=next_database_revision(db)
        wal_vectors=[@view batch.vectors[:,index] for index in 1:batch.count]
        wal_body=database_wal_put_body(revision,wal_vectors,batch.metadata,batch.ids)
        has_usable_base(db)||make_vector_store_writable!(db.vector_store)

        return commit_database_mutation!(db,revision,wal_body) do
            for index in 1:batch.count
                id=batch.ids[index]
                update_database_record!(db,id,@view(batch.vectors[:,index]),batch.metadata[index])
                maybe_inject_database_mutation_fault!(:during_apply)
            end

            db.live_count+=new_count
            return batch.ids
        end
    end
end

function update!(db::VectorDB,id;vector::Union{Nothing,AbstractVector{<:Real}}=nothing,metadata::Union{Nothing,NamedTuple}=nothing,)
    return with_database_write(db.database_lock) do
        validate_database_fast(db)
        vector===nothing&&metadata===nothing&&throw(ArgumentError("update requires a vector or metadata"))
        has_database_id(db,id)||throw(KeyError(id))
        converted=if vector===nothing
            nothing
        else
            convert_validated_vector(vector,db.dim,db.metric)
        end

        current=get_database_record(db,id)
        resolved_vector=converted===nothing ? current.vector : converted
        resolved_metadata=metadata===nothing ? current.metadata : metadata
        ensure_delta_search_capacity!(db,[id])
        revision=next_database_revision(db)
        wal_body=database_wal_put_body(revision,[resolved_vector],[resolved_metadata],[id])
        has_usable_base(db)||make_vector_store_writable!(db.vector_store)

        return commit_database_mutation!(db,revision,wal_body) do
            update_database_record!(db,id,resolved_vector,resolved_metadata)
            maybe_inject_database_mutation_fault!(:during_apply)
            return db
        end
    end
end

function Base.delete!(db::VectorDB,id)
    return with_database_write(db.database_lock) do
        validate_database_fast(db)
        has_database_id(db,id)||throw(KeyError(id))
        revision=next_database_revision(db)
        wal_body=database_wal_delete_body(revision,[id])
        has_usable_base(db)||make_vector_store_writable!(db.vector_store)

        return commit_database_mutation!(db,revision,wal_body) do
            delete_database_record!(db,id)
            maybe_inject_database_mutation_fault!(:during_apply)
            db.live_count-=1
            return db
        end
    end
end

function Base.delete!(db::VectorDB,ids::AbstractVector)
    return with_database_write(db.database_lock) do
        validate_database_fast(db)
        resolved_ids=collect(ids)
        isempty(resolved_ids)&&return db
        length(Set{Any}(resolved_ids))==length(resolved_ids)||throw(ArgumentError("batch ids must be unique"))

        for id in resolved_ids
            has_database_id(db,id)||throw(KeyError(id))
        end

        revision=next_database_revision(db)
        wal_body=database_wal_delete_body(revision,resolved_ids)
        has_usable_base(db)||make_vector_store_writable!(db.vector_store)

        return commit_database_mutation!(db,revision,wal_body) do
            for id in resolved_ids
                delete_database_record!(db,id)
                maybe_inject_database_mutation_fault!(:during_apply)
            end

            db.live_count-=length(resolved_ids)
            return db
        end
    end
end

function get_database_record(db::VectorDB,id)
    if has_delta_id(db.delta_store,id)
        position=get_position(db.delta_store.id_store,id)
        return(
            id=get_id(db.delta_store.id_store,position),
            vector=collect(get_vector(db.delta_store.vector_store,position)),
            metadata=get_metadata(db.delta_store.metadata_store,position),
        )
    end

    has_id(db.id_store,id)||throw(KeyError(id))
    position=get_position(db.id_store,id)
    db.base_tombstones[position]&&throw(KeyError(id))
    return(
        id=get_id(db.id_store,position),
        vector=collect(get_vector(db.vector_store,position)),
        metadata=get_metadata(db.metadata_store,position),
    )
end

function get_record(db::VectorDB,id)
    return with_database_read(db.database_lock) do
        validate_database_fast(db)
        return get_database_record(db,id)
    end
end

function materialize_database(db::VectorDB)
    count=logical_length(db)
    vector_store=create_vector_store(db.dim;initial_capacity=count,)
    metadata_store=create_metadata_store(initial_capacity=count,)
    id_store=create_id_store(initial_capacity=count,)

    for position in 1:length(db.vector_store)
        db.base_tombstones[position]&&continue
        insert_vector!(vector_store,get_vector(db.vector_store,position))
        insert_metadata!(metadata_store,get_metadata(db.metadata_store,position))
        insert_id!(id_store,get_id(db.id_store,position))
    end

    for position in 1:length(db.delta_store)
        insert_vector!(vector_store,get_vector(db.delta_store.vector_store,position))
        insert_metadata!(metadata_store,get_metadata(db.delta_store.metadata_store,position))
        insert_id!(id_store,get_id(db.delta_store.id_store,position))
    end

    length(vector_store)==count||error("materialized database count is misaligned")
    return(vector_store=vector_store,metadata_store=metadata_store,id_store=id_store,)
end

function build!(db::VectorDB;nlists::Int,iterations::Int=20,seed::Int=42,restarts::Int=1,training_count::Union{Nothing,Int}=nothing,_snapshot_hook::Union{Nothing,Function}=nothing,)
    snapshot=with_database_read(db.database_lock) do
        validate_database_fast(db)
        count=logical_length(db)
        count>0||throw(ArgumentError("database cannot be empty"))
        resolved_training_count=training_count===nothing ? count : training_count
        config=IndexBuildConfig(nlists,count;iterations=iterations,seed=seed,restarts=restarts,training_count=resolved_training_count,)
        stores=materialize_database(db)
        return(revision=db.revision,config=config,stores=stores,metric=db.metric,)
    end

    _snapshot_hook===nothing||_snapshot_hook(snapshot)

    vectors=stored_vectors(snapshot.stores.vector_store)
    metadata=stored_metadata(snapshot.stores.metadata_store)
    index=build_filter_aware_ivf(vectors,metadata;nlists=snapshot.config.nlists,iterations=snapshot.config.iterations,seed=snapshot.config.seed,metric=snapshot.metric,restarts=snapshot.config.restarts,training_count=snapshot.config.training_count,)
    filter_index=build_bitset_index(metadata)
    index_bytes=database_index_bytes(index,filter_index)
    built=(revision=snapshot.revision,config=snapshot.config,stores=snapshot.stores,index=index,filter_index=filter_index,index_bytes=index_bytes,)

    return install_database_index!(db,built)
end

function database_mutation_tail(db::VectorDB,revision::UInt64)
    revision<=db.revision||throw(ArgumentError("index build revision is ahead of database"))
    entries=DatabaseMutationEntry[entry for entry in db.mutation_history if entry.revision>revision]
    expected=revision

    for entry in entries
        expected==typemax(UInt64)&&throw(ArgumentError("mutation history revision overflow"))
        expected+=UInt64(1)
        entry.revision==expected||throw(ArgumentError("mutation history does not cover the index build tail"))
    end

    expected==db.revision||throw(ArgumentError("mutation history does not reach the current database revision"))
    return entries
end

function rebase_database_index(db::VectorDB,built,entries::Vector{DatabaseMutationEntry})
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
        checkpoint_operations=db.checkpoint_operations,
        checkpoint_bytes=db.checkpoint_bytes,
        checkpoint_retain_snapshots=db.checkpoint_retain_snapshots,
        maintenance_config=db.maintenance_config,
    )
    candidate.index_bytes=built.index_bytes

    for entry in entries
        apply_database_wal_record!(candidate,read_database_wal_record(entry.body,db.dim))
    end

    candidate.revision==db.revision||error("rebased index generation did not reach the current revision")
    candidate.live_count==db.live_count||error("rebased index generation changed the logical record count")
    delta_search_work(candidate)<=db.maintenance_config.max_delta_search_records||throw(ArgumentError("rebased index generation exceeds the configured delta search bound"))
    validate_database(candidate)
    return candidate
end

function install_database_index!(db::VectorDB,built)
    return with_database_write(db.database_lock) do
        all(property->hasproperty(built,property),(:revision,:config,:stores,:index,:filter_index,:index_bytes,))||throw(ArgumentError("index build generation is incomplete"))
        built.revision<=db.revision||throw(ArgumentError("index build revision is ahead of database"))
        db.index_revision!==nothing&&built.revision<db.index_revision&&return db
        entries=database_mutation_tail(db,built.revision)
        candidate=rebase_database_index(db,built,entries)
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
        db.build_config===nothing&&throw(ArgumentError("database has no previous index build configuration"))
        return db.build_config
    end
    count=with_database_read(db.database_lock) do
        logical_length(db)
    end
    config.nlists<=count||throw(ArgumentError("database has fewer vectors than its configured list count"))
    training_count=min(count,max(config.nlists,config.training_count))
    return build!(db;nlists=config.nlists,iterations=config.iterations,seed=config.seed,restarts=config.restarts,training_count=training_count,)
end

function database_search_results(raw_results::AbstractVector,id_store::IDStore;index_offset::Int=0,)
    results=Vector{SearchResult}(undef,length(raw_results))

    for index in eachindex(raw_results)
        result=raw_results[index]
        results[index]=SearchResult(
            get_id(id_store,result.index),
            index_offset+result.index,
            result.score,
            result.metadata,
        )
    end

    return results
end

function merge_database_search_results(base_results::Vector{SearchResult},delta_results::Vector{SearchResult},k::Int)
    isempty(delta_results)&&return base_results
    isempty(base_results)&&return delta_results
    count=min(k,length(base_results)+length(delta_results))
    results=Vector{SearchResult}(undef,count)
    base_position=1
    delta_position=1

    for position in 1:count
        take_base=if base_position>length(base_results)
            false
        elseif delta_position>length(delta_results)
            true
        else
            base_score=base_results[base_position].score
            delta_score=delta_results[delta_position].score
            isless(delta_score,base_score)||isequal(base_score,delta_score)
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

function search_database_locked(db::VectorDB,query::AbstractVector{<:Real};k::Int=10,nprobe::Union{Nothing,Int}=nothing,filter::Union{Nothing,FilterExpr}=nothing,strategy::Symbol=:auto,planner_config::PlannerConfig=PlannerConfig(),postfilter_oversample::Int=10,adaptive_postfilter::Bool=true,adaptive::Bool=true,max_nprobe::Union{Nothing,Int}=nothing,candidate_multiplier::Float64=planner_config.candidate_multiplier,postfilter_candidate_multiplier::Float64=planner_config.postfilter_candidate_multiplier,vector_weight::Float64=0.5,filter_weight::Float64=0.5,rerank_factor::Int=4,_plan::Union{Nothing,QueryPlan}=nothing,)
    length(query)==db.dim||throw(DimensionMismatch("query dimension doesnt match database"))
    k>0||throw(ArgumentError("k must be positive"))
    strategy===:auto||strategy_from_symbol(filter,strategy)
    base_vectors=stored_vectors(db.vector_store)
    base_metadata=stored_metadata(db.metadata_store)
    index=db.index

    if index===nothing
        strategy in (:auto,:exact)||throw(ArgumentError("search strategy $(strategy) requires a built index"))
        raw_results=search_exact(base_vectors,base_metadata,query;k=k,metric=db.metric,filter=filter,)
        return database_search_results(raw_results,db.id_store)
    end

    plan=_plan===nothing ? (strategy===:auto ? cached_plan_query(db,filter;k=k,config=planner_config,) : plan_query(db,filter;k=k,strategy=strategy,config=planner_config,)) : _plan
    resolved_minimum_nprobe=nprobe===nothing ? max(1,plan.minimum_nprobe) : nprobe
    resolved_nprobe=nprobe===nothing ? max(1,plan.nprobe) : nprobe
    list_count=length(index.ivf.lists)
    1<=resolved_nprobe<=list_count||throw(ArgumentError("nprobe must be between 1 and list count"))
    resolved_max_nprobe=max_nprobe===nothing ? resolved_nprobe : max_nprobe
    resolved_max_nprobe=clamp(resolved_max_nprobe,resolved_nprobe,list_count)

    raw_results=if plan.strategy isa ExactStrategy||plan.strategy isa PreFilterExactStrategy
        search_exact(base_vectors,base_metadata,query;k=k,metric=db.metric,filter=filter,filter_index=db.filter_index,vector_norms=index.ivf.vector_norms,excluded=db.base_tombstones,)
    elseif plan.strategy isa IVFStrategy
        search_ivf(index.ivf,base_vectors,base_metadata,query;k=k,nprobe=resolved_nprobe,metric=db.metric,excluded=db.base_tombstones,)
    elseif plan.strategy isa IVFPreFilterStrategy
        search_ivf_prefilter(index,base_vectors,base_metadata,query;k=k,nprobe=resolved_nprobe,metric=db.metric,filter=filter,excluded=db.base_tombstones,)
    elseif plan.strategy isa IVFPostFilterStrategy
        resolved_oversample=adaptive_postfilter ? resolve_postfilter_oversample(postfilter_oversample,plan.selectivity;candidate_multiplier=postfilter_candidate_multiplier,) : postfilter_oversample
        search_ivf_postfilter(index.ivf,base_vectors,base_metadata,query;k=k,nprobe=resolved_nprobe,metric=db.metric,filter=filter,oversample=resolved_oversample,excluded=db.base_tombstones,)
    elseif plan.strategy isa BoundedFilterAwareIVFStrategy
        search_filter_aware_bound(index,base_vectors,base_metadata,query;k=k,minimum_nprobe=resolved_minimum_nprobe,max_nprobe=resolved_max_nprobe,metric=db.metric,filter=filter,excluded=db.base_tombstones,)
    else
        minimum_nprobe=adaptive ? resolved_minimum_nprobe : resolved_nprobe
        search_filter_aware_ivf(index,base_vectors,base_metadata,query;k=k,nprobe=minimum_nprobe,metric=db.metric,filter=filter,adaptive=adaptive,max_nprobe=resolved_max_nprobe,candidate_multiplier=candidate_multiplier,vector_weight=vector_weight,filter_weight=filter_weight,rerank_factor=rerank_factor,excluded=db.base_tombstones,)
    end

    base_results=database_search_results(raw_results,db.id_store)
    delta_vectors=stored_vectors(db.delta_store.vector_store)
    delta_metadata=stored_metadata(db.delta_store.metadata_store)
    length(delta_metadata)<=db.maintenance_config.max_delta_search_records||error("delta search work exceeds the configured bound")
    delta_raw=search_exact(delta_vectors,delta_metadata,query;k=k,metric=db.metric,filter=filter,)
    delta_results=database_search_results(delta_raw,db.delta_store.id_store;index_offset=length(db.vector_store),)
    return merge_database_search_results(base_results,delta_results,k)
end

function search(db::VectorDB,query::AbstractVector{<:Real};filter::Union{Nothing,NamedTuple,FilterExpr}=nothing,kwargs...)
    normalized_filter=normalize_filter(filter)

    return with_database_read(db.database_lock) do
        validate_database_fast(db)
        converted=convert_validated_vector(query,db.dim,db.metric;context="query",)
        search_database_locked(db,converted;filter=normalized_filter,kwargs...)
    end
end

const DATABASE_BATCH_WORKER_KEY=:sen_database_batch_worker
const DEFAULT_BATCH_PARALLEL_THRESHOLD=4

function database_batch_worker_owned()
    return get(task_local_storage(),DATABASE_BATCH_WORKER_KEY,false)===true
end

function with_database_batch_worker(f::Function)
    storage=task_local_storage()
    had_owner=haskey(storage,DATABASE_BATCH_WORKER_KEY)
    previous_owner=get(storage,DATABASE_BATCH_WORKER_KEY,false)
    storage[DATABASE_BATCH_WORKER_KEY]=true

    try
        return f()
    finally
        if had_owner
            storage[DATABASE_BATCH_WORKER_KEY]=previous_owner
        else
            delete!(storage,DATABASE_BATCH_WORKER_KEY)
        end
    end
end

function resolve_database_batch_workers(query_count::Int;parallel::Bool,workers::Int,parallel_threshold::Int,)
    workers>0||throw(ArgumentError("batch workers must be positive"))
    parallel_threshold>0||throw(ArgumentError("batch parallel threshold must be positive"))

    if !parallel||query_count<parallel_threshold||database_batch_worker_owned()
        return 1
    end

    return max(1,min(query_count,workers,Threads.nthreads(:default)))
end

function resolve_database_batch_plan(db::VectorDB,filter::Union{Nothing,FilterExpr},kwargs)
    db.index===nothing&&return nothing
    k=get(kwargs,:k,10)
    strategy=get(kwargs,:strategy,:auto)
    planner_config=get(kwargs,:planner_config,PlannerConfig())
    return strategy===:auto ? cached_plan_query(db,filter;k=k,config=planner_config,) : plan_query(db,filter;k=k,strategy=strategy,config=planner_config,)
end

function search_database_exact_batch_locked(db::VectorDB,queries::AbstractMatrix;k::Int)
    base_vectors=stored_vectors(db.vector_store)
    base_metadata=stored_metadata(db.metadata_store)
    vector_norms=db.index===nothing ? nothing : db.index.ivf.vector_norms
    base_raw=search_exact_batch(base_vectors,base_metadata,queries;k=k,metric=db.metric,vector_norms=vector_norms,excluded=db.index===nothing ? nothing : db.base_tombstones,)
    delta_vectors=stored_vectors(db.delta_store.vector_store)
    delta_metadata=stored_metadata(db.delta_store.metadata_store)
    length(delta_metadata)<=db.maintenance_config.max_delta_search_records||error("delta search work exceeds the configured bound")
    delta_raw=search_exact_batch(delta_vectors,delta_metadata,queries;k=k,metric=db.metric,)
    results=Vector{Vector{SearchResult}}(undef,size(queries,2))

    for query_index in axes(queries,2)
        base_results=database_search_results(base_raw[query_index],db.id_store)
        delta_results=database_search_results(delta_raw[query_index],db.delta_store.id_store;index_offset=length(db.vector_store),)
        results[query_index]=merge_database_search_results(base_results,delta_results,k)
    end

    return results
end

function use_exact_batch_search(db::VectorDB,filter::Union{Nothing,FilterExpr},plan,kwargs)
    filter===nothing||return false
    haskey(kwargs,:nprobe)&&return false
    haskey(kwargs,:max_nprobe)&&return false
    strategy=get(kwargs,:strategy,:auto)
    db.index===nothing&&return strategy in (:auto,:exact)
    return plan!==nothing&&plan.strategy isa ExactStrategy
end

function search_database_batch_parallel!(results::Vector{Vector{SearchResult}},db::VectorDB,queries::AbstractMatrix{<:Real},worker_count::Int;kwargs...)
    query_count=size(queries,2)
    jobs=Channel{Int}(query_count)

    for index in 1:query_count
        put!(jobs,index)
    end

    close(jobs)
    failures=Vector{Union{Nothing,Tuple{Int,Exception}}}(nothing,worker_count)
    tasks=Vector{Task}(undef,worker_count)

    for worker_index in 1:worker_count
        tasks[worker_index]=Threads.@spawn with_database_batch_worker() do
            query_index=0

            try
                for index in jobs
                    query_index=index
                    results[index]=search_database_locked(db,@view(queries[:,index]);kwargs...)
                end
            catch error
                failures[worker_index]=(query_index,error)
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

function search(db::VectorDB,queries::AbstractMatrix{<:Real};filter::Union{Nothing,NamedTuple,FilterExpr}=nothing,parallel::Bool=true,workers::Int=Threads.nthreads(:default),parallel_threshold::Int=DEFAULT_BATCH_PARALLEL_THRESHOLD,kwargs...)
    normalized_filter=normalize_filter(filter)

    return with_database_read(db.database_lock) do
        validate_database_fast(db)
        converted=convert_validated_vectors(queries,db.dim,db.metric;context="query",)
        query_count=size(converted,2)
        results=Vector{Vector{SearchResult}}(undef,query_count)
        worker_count=resolve_database_batch_workers(query_count;parallel=parallel,workers=workers,parallel_threshold=parallel_threshold,)
        query_count==0&&return results
        batch_plan=resolve_database_batch_plan(db,normalized_filter,kwargs)

        if use_exact_batch_search(db,normalized_filter,batch_plan,kwargs)
            return search_database_exact_batch_locked(db,converted;k=get(kwargs,:k,10),)
        end

        if worker_count==1
            for index in 1:query_count
                results[index]=search_database_locked(db,@view(converted[:,index]);filter=normalized_filter,_plan=batch_plan,kwargs...)
            end

            return results
        end

        return search_database_batch_parallel!(results,db,converted,worker_count;filter=normalized_filter,_plan=batch_plan,kwargs...)
    end
end

function save!(db::VectorDB;retain_snapshots::Int=2,)
    return with_database_write(db.database_lock) do
        activate_database_writer_lock!(db)
        validate_database(db)
        retain_snapshots>0||throw(ArgumentError("retain snapshots must be positive"))
        config=db.build_config
        current_index=has_usable_base(db)&&db.index_revision==db.revision
        stores=current_index ? (vector_store=db.vector_store,metadata_store=db.metadata_store,id_store=db.id_store,) : materialize_database(db)
        count=length(stores.vector_store)
        persisted_nlists=config===nothing||count==0 ? nothing : min(config.nlists,count)
        persisted_training_count=persisted_nlists===nothing ? count : min(count,max(persisted_nlists,config.training_count))
        manifest=create_database_manifest(
            db.dim,
            db.metric,
            count;
            format_version=2,
            nlists=persisted_nlists,
            revision=db.revision,
            index_revision=current_index ? db.index_revision : nothing,
            iterations=config===nothing ? 20 : config.iterations,
            seed=config===nothing ? 42 : config.seed,
            restarts=config===nothing ? 1 : config.restarts,
            training_count=persisted_training_count,
        )
        snapshot=new_database_snapshot(db.path)

        try
            save_manifest(snapshot.temporary_path,manifest)
            save_vector_store(snapshot.temporary_path,stores.vector_store)
            save_metadata_store(snapshot.temporary_path,stores.metadata_store)
            save_id_store(snapshot.temporary_path,stores.id_store)
            current_index&&save_ivf_index(snapshot.temporary_path,db.index.ivf)
            seal_database_snapshot(snapshot.temporary_path,db.revision)
            maybe_pause_database_process!(:after_snapshot_seal,db.path)
            commit_database_snapshot(db.path,snapshot)
            maybe_pause_database_process!(:after_snapshot_commit,db.path)
            checkpoint_database_wal!(db)
        catch
            abort_database_snapshot(snapshot)
            rethrow()
        end

        prune_database_snapshots(db.path;retain=retain_snapshots,)
        return db
    end
end

function load_db_from_snapshot(path::AbstractString,snapshot_path::AbstractString;iterations::Int=20,seed::Int=42,rebuild::Bool=false,checkpoint_operations::Int=10_000,checkpoint_bytes::Int=64*1024*1024,checkpoint_retain_snapshots::Int=2,maintenance_config::MaintenanceConfig=MaintenanceConfig(),mmap_vectors::Union{Bool,Symbol}=:auto,mmap_threshold_bytes::Int=DEFAULT_VECTOR_MMAP_THRESHOLD_BYTES,)
    descriptor=validate_database_snapshot(snapshot_path)
    manifest=load_manifest(snapshot_path)
    descriptor.revision===nothing||descriptor.revision==manifest.revision||throw(ArgumentError("snapshot revision doesnt match manifest"))
    vector_store=load_vector_store(snapshot_path;mmap=mmap_vectors,mmap_threshold_bytes=mmap_threshold_bytes,)
    metadata_store=load_metadata_store(snapshot_path)
    id_store=load_id_store(snapshot_path)

    vector_store.dim==manifest.dim||throw(DimensionMismatch("stored vector dimension doesnt match manifest"))
    length(vector_store)==manifest.count||throw(DimensionMismatch("stored vector count doesnt match manifest"))
    length(metadata_store)==manifest.count||throw(DimensionMismatch("stored metadata count doesnt match manifest"))
    length(id_store)==manifest.count||throw(DimensionMismatch("stored id count doesnt match manifest"))
    validate_stored_vectors!(stored_vectors(vector_store),manifest.metric)

    db=VectorDB(
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
        checkpoint_operations=checkpoint_operations,
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_retain_snapshots=checkpoint_retain_snapshots,
        maintenance_config=maintenance_config,
    )

    if manifest.index_revision==manifest.revision&&isfile(index_file_path(snapshot_path))
        ivf=load_ivf_index(snapshot_path)
        size(ivf.centroids,1)==manifest.dim||throw(DimensionMismatch("stored index dimension doesnt match manifest"))
        manifest.nlists===nothing||length(ivf.lists)==manifest.nlists||throw(DimensionMismatch("stored list count doesnt match manifest"))
        sum(length,ivf.lists)==manifest.count||throw(DimensionMismatch("stored index count doesnt match manifest"))
        ivf.metric===manifest.metric||throw(ArgumentError("stored index metric doesnt match manifest"))
        db.index=build_filter_aware_ivf(ivf,stored_metadata(metadata_store))
        db.filter_index=build_bitset_index(stored_metadata(metadata_store))
        db.index_revision=manifest.index_revision
        db.index_bytes=database_index_bytes(db.index,db.filter_index)
    elseif rebuild&&manifest.build_config!==nothing
        config=manifest.build_config
        build!(db;nlists=config.nlists,iterations=config.iterations,seed=config.seed,restarts=config.restarts,training_count=min(length(db),max(config.nlists,config.training_count)),)
    end

    replay_database_wal!(db)
    ensure_delta_search_capacity!(db,Any[])
    validate_database(db)
    return db
end

function load_db_from_wal(path::AbstractString;checkpoint_operations::Int=10_000,checkpoint_bytes::Int=64*1024*1024,checkpoint_retain_snapshots::Int=2,maintenance_config::MaintenanceConfig=MaintenanceConfig(),)
    wal=read_database_wal(path;repair_tail=true,)
    wal===nothing&&throw(ArgumentError("database WAL does not exist"))
    wal.header.revision==UInt64(0)||throw(ArgumentError("database snapshot is missing for checkpointed WAL"))
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
        checkpoint_operations=checkpoint_operations,
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_retain_snapshots=checkpoint_retain_snapshots,
        maintenance_config=maintenance_config,
    )
    replay_database_wal!(db)
    ensure_delta_search_capacity!(db,Any[])
    validate_database(db)
    return db
end

function rebase_recovered_database_wal!(db::VectorDB)
    checkpoint_revision=db.wal_checkpoint_revision
    checkpoint_revision===nothing&&return db
    checkpoint_revision<=db.revision&&return db
    replace_database_wal(db.path,db.revision,db.dim,db.metric)
    db.wal_revision=db.revision
    db.wal_checkpoint_revision=db.revision
    return db
end

function load_db_without_writer_lock(path::AbstractString;iterations::Int=20,seed::Int=42,rebuild::Bool=false,recover::Bool=false,checkpoint_operations::Int=10_000,checkpoint_bytes::Int=64*1024*1024,checkpoint_retain_snapshots::Int=2,maintenance_config::MaintenanceConfig=MaintenanceConfig(),mmap_vectors::Union{Bool,Symbol}=:auto,mmap_threshold_bytes::Int=DEFAULT_VECTOR_MMAP_THRESHOLD_BYTES,)
    if isfile(database_wal_path(path))&&!isfile(database_current_path(path))&&!isfile(joinpath(path,"manifest.toml"))&&isempty(database_snapshot_generations(path))
        return load_db_from_wal(path;checkpoint_operations=checkpoint_operations,checkpoint_bytes=checkpoint_bytes,checkpoint_retain_snapshots=checkpoint_retain_snapshots,maintenance_config=maintenance_config,)
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
        db=load_db_from_snapshot(path,snapshot_path;iterations=iterations,seed=seed,rebuild=rebuild,checkpoint_operations=checkpoint_operations,checkpoint_bytes=checkpoint_bytes,checkpoint_retain_snapshots=checkpoint_retain_snapshots,maintenance_config=maintenance_config,mmap_vectors=mmap_vectors,mmap_threshold_bytes=mmap_threshold_bytes,)
        recovered_snapshot&&rebase_recovered_database_wal!(db)
        return db
    catch
        recover||rethrow()
        recovered_path=recover_database_snapshot(path)
        db=load_db_from_snapshot(path,recovered_path;iterations=iterations,seed=seed,rebuild=rebuild,checkpoint_operations=checkpoint_operations,checkpoint_bytes=checkpoint_bytes,checkpoint_retain_snapshots=checkpoint_retain_snapshots,maintenance_config=maintenance_config,mmap_vectors=mmap_vectors,mmap_threshold_bytes=mmap_threshold_bytes,)
        rebase_recovered_database_wal!(db)
        return db
    end
end

function load_db(path::AbstractString;iterations::Int=20,seed::Int=42,rebuild::Bool=false,recover::Bool=false,checkpoint_operations::Int=10_000,checkpoint_bytes::Int=64*1024*1024,checkpoint_retain_snapshots::Int=2,maintenance_config::MaintenanceConfig=MaintenanceConfig(),mmap_vectors::Union{Bool,Symbol}=:auto,mmap_threshold_bytes::Int=DEFAULT_VECTOR_MMAP_THRESHOLD_BYTES,)
    io=acquire_database_writer_lock(path)

    try
        db=load_db_without_writer_lock(path;iterations=iterations,seed=seed,rebuild=rebuild,recover=recover,checkpoint_operations=checkpoint_operations,checkpoint_bytes=checkpoint_bytes,checkpoint_retain_snapshots=checkpoint_retain_snapshots,maintenance_config=maintenance_config,mmap_vectors=mmap_vectors,mmap_threshold_bytes=mmap_threshold_bytes,)
        attach_database_writer_lock!(db,io)
        maybe_schedule_database_maintenance!(db)
        return db
    catch
        release_database_writer_lock(io)
        rethrow()
    end
end

function recover_db(path::AbstractString;iterations::Int=20,seed::Int=42,rebuild::Bool=false,checkpoint_operations::Int=10_000,checkpoint_bytes::Int=64*1024*1024,checkpoint_retain_snapshots::Int=2,maintenance_config::MaintenanceConfig=MaintenanceConfig(),mmap_vectors::Union{Bool,Symbol}=:auto,mmap_threshold_bytes::Int=DEFAULT_VECTOR_MMAP_THRESHOLD_BYTES,)
    return load_db(path;iterations=iterations,seed=seed,rebuild=rebuild,recover=true,checkpoint_operations=checkpoint_operations,checkpoint_bytes=checkpoint_bytes,checkpoint_retain_snapshots=checkpoint_retain_snapshots,maintenance_config=maintenance_config,mmap_vectors=mmap_vectors,mmap_threshold_bytes=mmap_threshold_bytes,)
end
