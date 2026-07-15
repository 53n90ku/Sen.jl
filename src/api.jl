function create_db(path::AbstractString;dim::Int,metric::Symbol=:cosine,initial_capacity::Int=0,durable::Bool=true,checkpoint_operations::Int=10_000,checkpoint_bytes::Int=64*1024*1024,checkpoint_retain_snapshots::Int=2,)
    dim>0||throw(ArgumentError("dimension must be positive"))
    metric in (:cosine,:dot)||throw(ArgumentError("metric must be :cosine or :dot"))
    initial_capacity>=0||throw(ArgumentError("initial capacity cannot be negative"))

    vector_store=create_vector_store(dim;initial_capacity=initial_capacity,)
    metadata_store=create_metadata_store(initial_capacity=initial_capacity,)
    id_store=create_id_store(initial_capacity=initial_capacity,)

    db=VectorDB(String(path),dim,metric,vector_store,metadata_store,id_store,nothing,nothing,nothing,UInt64(0),nothing,DatabaseLock(),Dict{Any,Any}(),ReentrantLock();checkpoint_operations=checkpoint_operations,checkpoint_bytes=checkpoint_bytes,checkpoint_retain_snapshots=checkpoint_retain_snapshots,)

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
        isempty(db.delta_store.id_store.ids)||error("unindexed database has delta records")
        db.live_count==count||error("unindexed database live count is misaligned")
    else
        db.filter_index===nothing&&error("database vector index exists without filter index")
        db.index_revision===nothing&&error("database vector index has no revision")
        db.index_revision<=db.revision||error("database index revision is ahead of database")
        sum(length,db.index.ivf.lists)==count||error("database index count is misaligned")
        db.filter_index.count==count||error("database filter index count is misaligned")
    end

    return db
end

function validate_database(db::VectorDB)
    validate_database_fast(db)

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

function finish_database_mutation!(db::VectorDB,revision::UInt64)
    revision==next_database_revision(db)||error("database mutation revision is invalid")
    db.revision=revision
    clear_plan_cache!(db)
    validate_database_fast(db)
    maybe_checkpoint_database!(db)
    return db
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
        length(vector)==db.dim||throw(DimensionMismatch("vector dimension doesnt match database"))
        converted=Float32[value for value in vector]
        resolved_id=id===nothing ? next_database_id(db) : id
        resolved_id===nothing&&throw(ArgumentError("id cannot be nothing"))
        has_database_id(db,resolved_id)&&throw(ArgumentError("id already exists"))
        revision=next_database_revision(db)
        append_database_wal_put!(db,revision,[converted],[metadata],[resolved_id])

        if has_usable_base(db)
            insert_delta!(db.delta_store,converted,metadata,resolved_id)
        else
            vector_position=insert_vector!(db.vector_store,converted)
            metadata_position=insert_metadata!(db.metadata_store,metadata)
            id_position=insert_id!(db.id_store,resolved_id)
            vector_position==metadata_position==id_position||error("database stores are misaligned")
            push!(db.base_tombstones,false)
        end

        db.live_count+=1
        finish_database_mutation!(db,revision)
        return resolved_id
    end
end

function prepare_database_batch(db::VectorDB,vectors::AbstractMatrix,metadata::AbstractVector{<:NamedTuple},ids;allow_existing::Bool,)
    dim,count=size(vectors)
    dim==db.dim||throw(DimensionMismatch("vector dimensions dont match database"))
    length(metadata)==count||throw(DimensionMismatch("metadata count doesnt match vector count"))
    converted=Matrix{Float32}(vectors)
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
        revision=next_database_revision(db)
        wal_vectors=[@view batch.vectors[:,index] for index in 1:batch.count]
        append_database_wal_put!(db,revision,wal_vectors,batch.metadata,batch.ids)

        for index in 1:batch.count
            insert_database_record!(db,@view(batch.vectors[:,index]),batch.metadata[index],batch.ids[index])
        end

        db.live_count+=batch.count
        finish_database_mutation!(db,revision)
        return batch.ids
    end
end

function upsert!(db::VectorDB,vector::AbstractVector{<:Real},metadata::NamedTuple;id,)
    return with_database_write(db.database_lock) do
        validate_database_fast(db)
        id===nothing&&throw(ArgumentError("id cannot be nothing"))
        length(vector)==db.dim||throw(DimensionMismatch("vector dimension doesnt match database"))
        converted=Float32[value for value in vector]
        existing=has_database_id(db,id)
        revision=next_database_revision(db)
        append_database_wal_put!(db,revision,[converted],[metadata],[id])

        update_database_record!(db,id,converted,metadata)
        existing||(db.live_count+=1)
        finish_database_mutation!(db,revision)
        return id
    end
end

function upsert!(db::VectorDB,vectors::AbstractMatrix,metadata::AbstractVector{<:NamedTuple};ids,)
    return with_database_write(db.database_lock) do
        validate_database_fast(db)
        batch=prepare_database_batch(db,vectors,metadata,ids;allow_existing=true,)
        batch.count==0&&return batch.ids
        new_count=count(id->!has_database_id(db,id),batch.ids)
        revision=next_database_revision(db)
        wal_vectors=[@view batch.vectors[:,index] for index in 1:batch.count]
        append_database_wal_put!(db,revision,wal_vectors,batch.metadata,batch.ids)

        for index in 1:batch.count
            id=batch.ids[index]
            update_database_record!(db,id,@view(batch.vectors[:,index]),batch.metadata[index])
        end

        db.live_count+=new_count
        finish_database_mutation!(db,revision)
        return batch.ids
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
            length(vector)==db.dim||throw(DimensionMismatch("vector dimension doesnt match database"))
            Float32[value for value in vector]
        end

        current=get_database_record(db,id)
        resolved_vector=converted===nothing ? current.vector : converted
        resolved_metadata=metadata===nothing ? current.metadata : metadata
        revision=next_database_revision(db)
        append_database_wal_put!(db,revision,[resolved_vector],[resolved_metadata],[id])
        update_database_record!(db,id,resolved_vector,resolved_metadata)
        finish_database_mutation!(db,revision)
        return db
    end
end

function Base.delete!(db::VectorDB,id)
    return with_database_write(db.database_lock) do
        validate_database_fast(db)
        has_database_id(db,id)||throw(KeyError(id))
        revision=next_database_revision(db)
        append_database_wal_delete!(db,revision,[id])
        delete_database_record!(db,id)
        db.live_count-=1
        finish_database_mutation!(db,revision)
        return db
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
        append_database_wal_delete!(db,revision,resolved_ids)

        for id in resolved_ids
            delete_database_record!(db,id)
        end

        db.live_count-=length(resolved_ids)
        finish_database_mutation!(db,revision)
        return db
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

function build!(db::VectorDB;nlists::Int,iterations::Int=20,seed::Int=42,restarts::Int=1,training_count::Union{Nothing,Int}=nothing,)
    snapshot=with_database_read(db.database_lock) do
        validate_database_fast(db)
        count=logical_length(db)
        count>0||throw(ArgumentError("database cannot be empty"))
        resolved_training_count=training_count===nothing ? count : training_count
        config=IndexBuildConfig(nlists,count;iterations=iterations,seed=seed,restarts=restarts,training_count=resolved_training_count,)
        stores=materialize_database(db)
        return(revision=db.revision,config=config,stores=stores,metric=db.metric,)
    end

    vectors=stored_vectors(snapshot.stores.vector_store)
    metadata=stored_metadata(snapshot.stores.metadata_store)
    index=build_filter_aware_ivf(vectors,metadata;nlists=snapshot.config.nlists,iterations=snapshot.config.iterations,seed=snapshot.config.seed,metric=snapshot.metric,restarts=snapshot.config.restarts,training_count=snapshot.config.training_count,)
    filter_index=build_bitset_index(metadata)
    built=(revision=snapshot.revision,config=snapshot.config,stores=snapshot.stores,index=index,filter_index=filter_index,)

    return install_database_index!(db,built)
end

function install_database_index!(db::VectorDB,built)
    return with_database_write(db.database_lock) do
        db.revision==built.revision||throw(ArgumentError("database changed while index was building"))
        db.vector_store=built.stores.vector_store
        db.metadata_store=built.stores.metadata_store
        db.id_store=built.stores.id_store
        db.index=built.index
        db.filter_index=built.filter_index
        db.build_config=built.config
        db.index_revision=built.revision
        db.delta_store=create_delta_store(db.dim)
        db.base_tombstones=falses(length(db.vector_store))
        db.live_count=length(db.vector_store)
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
    results=vcat(base_results,delta_results)
    sort!(results;by=result->result.score,rev=true,)
    resize!(results,min(k,length(results)))
    return results
end

function search(db::VectorDB,query::AbstractVector{<:Real};k::Int=10,nprobe::Union{Nothing,Int}=nothing,filter::Union{Nothing,NamedTuple}=nothing,strategy::Symbol=:auto,planner_config::PlannerConfig=PlannerConfig(),postfilter_oversample::Int=10,adaptive_postfilter::Bool=true,adaptive::Bool=true,max_nprobe::Union{Nothing,Int}=nothing,candidate_multiplier::Float64=planner_config.candidate_multiplier,postfilter_candidate_multiplier::Float64=planner_config.postfilter_candidate_multiplier,vector_weight::Float64=0.5,filter_weight::Float64=0.5,rerank_factor::Int=4,)
    return with_database_read(db.database_lock) do
        validate_database_fast(db)
        length(query)==db.dim||throw(DimensionMismatch("query dimension doesnt match database"))
        k>0||throw(ArgumentError("k must be positive"))
        strategy===:auto||strategy_from_symbol(filter,strategy)
        base_vectors=stored_vectors(db.vector_store)
        base_metadata=stored_metadata(db.metadata_store)
        index=db.index

        if index===nothing
            raw_results=search_exact(base_vectors,base_metadata,query;k=k,metric=db.metric,filter=filter,)
            return database_search_results(raw_results,db.id_store)
        end

        plan=strategy===:auto ? cached_plan_query(db,filter;k=k,config=planner_config,) : plan_query(db,filter;k=k,strategy=strategy,config=planner_config,)
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
        delta_raw=search_exact(delta_vectors,delta_metadata,query;k=k,metric=db.metric,filter=filter,)
        delta_results=database_search_results(delta_raw,db.delta_store.id_store;index_offset=length(db.vector_store),)
        return merge_database_search_results(base_results,delta_results,k)
    end
end

function search(db::VectorDB,queries::AbstractMatrix{<:Real};kwargs...)
    return with_database_read(db.database_lock) do
        size(queries,1)==db.dim||throw(DimensionMismatch("query dimensions dont match database"))
        query_count=size(queries,2)
        results=Vector{Vector{SearchResult}}(undef,query_count)

        for index in 1:query_count
            results[index]=search(db,@view queries[:,index];kwargs...)
        end

        return results
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
            commit_database_snapshot(db.path,snapshot)
            checkpoint_database_wal!(db)
        catch
            abort_database_snapshot(snapshot)
            rethrow()
        end

        prune_database_snapshots(db.path;retain=retain_snapshots,)
        return db
    end
end

function load_db_from_snapshot(path::AbstractString,snapshot_path::AbstractString;iterations::Int=20,seed::Int=42,rebuild::Bool=false,checkpoint_operations::Int=10_000,checkpoint_bytes::Int=64*1024*1024,checkpoint_retain_snapshots::Int=2,)
    descriptor=validate_database_snapshot(snapshot_path)
    manifest=load_manifest(snapshot_path)
    descriptor.revision===nothing||descriptor.revision==manifest.revision||throw(ArgumentError("snapshot revision doesnt match manifest"))
    vector_store=load_vector_store(snapshot_path)
    metadata_store=load_metadata_store(snapshot_path)
    id_store=load_id_store(snapshot_path)

    vector_store.dim==manifest.dim||throw(DimensionMismatch("stored vector dimension doesnt match manifest"))
    length(vector_store)==manifest.count||throw(DimensionMismatch("stored vector count doesnt match manifest"))
    length(metadata_store)==manifest.count||throw(DimensionMismatch("stored metadata count doesnt match manifest"))
    length(id_store)==manifest.count||throw(DimensionMismatch("stored id count doesnt match manifest"))

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
    elseif rebuild&&manifest.build_config!==nothing
        config=manifest.build_config
        build!(db;nlists=config.nlists,iterations=config.iterations,seed=config.seed,restarts=config.restarts,training_count=min(length(db),max(config.nlists,config.training_count)),)
    end

    replay_database_wal!(db)
    validate_database(db)
    return db
end

function load_db_from_wal(path::AbstractString;checkpoint_operations::Int=10_000,checkpoint_bytes::Int=64*1024*1024,checkpoint_retain_snapshots::Int=2,)
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
    )
    replay_database_wal!(db)
    validate_database(db)
    return db
end

function load_db_without_writer_lock(path::AbstractString;iterations::Int=20,seed::Int=42,rebuild::Bool=false,recover::Bool=false,checkpoint_operations::Int=10_000,checkpoint_bytes::Int=64*1024*1024,checkpoint_retain_snapshots::Int=2,)
    if isfile(database_wal_path(path))&&!isfile(database_current_path(path))&&!isfile(joinpath(path,"manifest.toml"))&&isempty(database_snapshot_generations(path))
        return load_db_from_wal(path;checkpoint_operations=checkpoint_operations,checkpoint_bytes=checkpoint_bytes,checkpoint_retain_snapshots=checkpoint_retain_snapshots,)
    end

    snapshot_path=try
        current_database_snapshot(path)
    catch
        recover||rethrow()
        recover_database_snapshot(path)
    end

    try
        return load_db_from_snapshot(path,snapshot_path;iterations=iterations,seed=seed,rebuild=rebuild,checkpoint_operations=checkpoint_operations,checkpoint_bytes=checkpoint_bytes,checkpoint_retain_snapshots=checkpoint_retain_snapshots,)
    catch
        recover||rethrow()
        recovered_path=recover_database_snapshot(path)
        return load_db_from_snapshot(path,recovered_path;iterations=iterations,seed=seed,rebuild=rebuild,checkpoint_operations=checkpoint_operations,checkpoint_bytes=checkpoint_bytes,checkpoint_retain_snapshots=checkpoint_retain_snapshots,)
    end
end

function load_db(path::AbstractString;iterations::Int=20,seed::Int=42,rebuild::Bool=false,recover::Bool=false,checkpoint_operations::Int=10_000,checkpoint_bytes::Int=64*1024*1024,checkpoint_retain_snapshots::Int=2,)
    io=acquire_database_writer_lock(path)

    try
        db=load_db_without_writer_lock(path;iterations=iterations,seed=seed,rebuild=rebuild,recover=recover,checkpoint_operations=checkpoint_operations,checkpoint_bytes=checkpoint_bytes,checkpoint_retain_snapshots=checkpoint_retain_snapshots,)
        attach_database_writer_lock!(db,io)
        return db
    catch
        release_database_writer_lock(io)
        rethrow()
    end
end

function recover_db(path::AbstractString;iterations::Int=20,seed::Int=42,rebuild::Bool=false,checkpoint_operations::Int=10_000,checkpoint_bytes::Int=64*1024*1024,checkpoint_retain_snapshots::Int=2,)
    return load_db(path;iterations=iterations,seed=seed,rebuild=rebuild,recover=true,checkpoint_operations=checkpoint_operations,checkpoint_bytes=checkpoint_bytes,checkpoint_retain_snapshots=checkpoint_retain_snapshots,)
end
