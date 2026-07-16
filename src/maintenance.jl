function maintenance_counts_locked(db::VectorDB)
    base_count=length(db.vector_store)
    delta_count=length(db.delta_store)
    segment_count=length(db.immutable_segments)
    additional_tombstones=sum(
        segment->count(segment.excluded)+length(segment.tombstone_ids),
        Iterators.drop(db.immutable_segments, 1);
        init = 0,
    )
    active_additional_tombstones=count(
        id->!has_id(db.id_store, id),
        db.active_segment.tombstone_ids,
    )
    tombstone_count=count(db.base_tombstones)+additional_tombstones+active_additional_tombstones
    denominator=max(1, base_count)
    return (
        base_count = base_count,
        delta_count = delta_count,
        segment_count = segment_count,
        delta_search_work = delta_count,
        delta_search_limit = db.maintenance_config.max_delta_search_records,
        tombstone_count = tombstone_count,
        delta_ratio = delta_count/denominator,
        tombstone_ratio = tombstone_count/denominator,
    )
end

function database_maintenance_due_locked(db::VectorDB)
    config=db.maintenance_config
    config.enabled||return false
    db.closed&&return false
    db.index===nothing&&return false
    db.build_config===nothing&&return false
    db.live_count>0||return false
    counts=maintenance_counts_locked(db)
    segment_due=config.segment_compaction_threshold>0&&counts.segment_count>=config.segment_compaction_threshold
    segment_due&&return true
    counts.delta_count+counts.tombstone_count>=config.minimum_changes||return false
    delta_due=(config.delta_threshold>0&&counts.delta_count>=config.delta_threshold)||(
        config.delta_ratio>0&&counts.delta_ratio>=config.delta_ratio
    )
    tombstone_due=(
        config.tombstone_threshold>0&&counts.tombstone_count>=config.tombstone_threshold
    )||(config.tombstone_ratio>0&&counts.tombstone_ratio>=config.tombstone_ratio)
    return delta_due||tombstone_due
end

function database_maintenance_due(db::VectorDB)
    return with_database_read(db.database_lock) do
        database_maintenance_due_locked(db)
    end
end

function maintenance_stop_requested(db::VectorDB)
    state=db.maintenance_state
    lock(state.lock)

    try
        return state.stop_requested
    finally
        unlock(state.lock)
    end
end

function update_maintenance_state!(
    db::VectorDB;
    status::Union{Nothing,Symbol} = nothing,
    attempts::Union{Nothing,Int} = nothing,
    started_revision = nothing,
    completed_revision = nothing,
    duration_ms::Union{Nothing,Float64} = nothing,
    error_message = nothing,
    clear_error::Bool = false,
)
    state=db.maintenance_state
    lock(state.lock)

    try
        status===nothing||(state.status=status)
        attempts===nothing||(state.attempts=attempts)
        started_revision===nothing||(state.last_started_revision=started_revision)
        completed_revision===nothing||(state.last_completed_revision=completed_revision)
        duration_ms===nothing||(state.last_duration_ms=duration_ms)
        clear_error&&(state.last_error=nothing)
        error_message===nothing||(state.last_error=String(error_message))
    finally
        unlock(state.lock)
    end

    return db
end

function persist_maintenance_rebuild!(db::VectorDB)
    config=db.maintenance_config
    config.persist_after_rebuild||return nothing
    durable=with_database_read(db.database_lock) do
        !db.closed&&db.writer_lock!==nothing
    end
    durable||return nothing

    try
        save!(db; retain_snapshots = db.checkpoint_retain_snapshots)
        return nothing
    catch error
        error isa InterruptException&&rethrow()
        message=sprint(showerror, error)
        update_maintenance_state!(
            db;
            error_message = "index rebuilt but automatic snapshot failed: $(message)",
        )
        @warn "automatic maintenance snapshot failed; rebuilt index remains available and WAL remains durable" exception=(
            error,
            catch_backtrace(),
        )
        return nothing
    end
end

function rebuild_database_maintenance!(db::VectorDB)
    build=with_database_read(db.database_lock) do
        config=db.build_config
        config===nothing&&throw(
            ArgumentError("database has no previous index build configuration"),
        )
        count=logical_length(db)
        count>0||throw(ArgumentError("database cannot be empty"))
        nlists=min(config.nlists, count)
        training_count=min(count, max(nlists, config.training_count))
        return (
            nlists = nlists,
            iterations = config.iterations,
            seed = config.seed,
            restarts = config.restarts,
            training_count = training_count,
        )
    end

    return build!(db; build...)
end

function database_maintenance_worker!(db::VectorDB)
    state=db.maintenance_state
    config=db.maintenance_config
    started_ns=time_ns()

    try
        for attempt = 1:(config.max_retries+1)
            if maintenance_stop_requested(db)
                update_maintenance_state!(
                    db;
                    status = :stopped,
                    duration_ms = (time_ns()-started_ns)/1_000_000,
                )
                return nothing
            end

            if !database_maintenance_due(db)
                update_maintenance_state!(
                    db;
                    status = :idle,
                    duration_ms = (time_ns()-started_ns)/1_000_000,
                )
                return nothing
            end

            revision=with_database_read(db.database_lock) do
                db.revision
            end
            update_maintenance_state!(
                db;
                status = :building,
                attempts = attempt,
                started_revision = revision,
            )

            try
                rebuild_database_maintenance!(db)
                completed_revision=with_database_read(db.database_lock) do
                    db.index_revision
                end
                update_maintenance_state!(
                    db;
                    status = :completed,
                    completed_revision = completed_revision,
                    duration_ms = (time_ns()-started_ns)/1_000_000,
                    clear_error = true,
                )
                persist_maintenance_rebuild!(db)
                return nothing
            catch error
                error isa InterruptException&&rethrow()
                message=sprint(showerror, error)
                final_attempt=attempt>config.max_retries
                update_maintenance_state!(
                    db;
                    status = final_attempt ? :failed : :retrying,
                    duration_ms = (time_ns()-started_ns)/1_000_000,
                    error_message = message,
                )

                if final_attempt
                    @warn "automatic database maintenance failed; current base and delta remain searchable" exception=(
                        error,
                        catch_backtrace(),
                    )
                    return nothing
                end

                config.retry_delay_ms>0&&sleep(config.retry_delay_ms/1_000)
            end
        end
    finally
        lock(state.lock)
        pending=false
        stopped=false

        try
            pending=state.pending
            stopped=state.stop_requested
            state.pending=false
            state.task=nothing
        finally
            unlock(state.lock)
        end

        pending&&!stopped&&maybe_schedule_database_maintenance!(db)
    end

    return nothing
end

function maybe_schedule_database_maintenance!(db::VectorDB)
    database_maintenance_due(db)||return false
    state=db.maintenance_state
    lock(state.lock)

    try
        state.stop_requested&&return false

        if state.task!==nothing
            state.pending=true
            return false
        end

        state.status=:scheduled
        state.pending=false
        state.task=Threads.@spawn database_maintenance_worker!(db)
        return true
    finally
        unlock(state.lock)
    end
end

function segment_indexing_due_locked(db::VectorDB)
    db.maintenance_config.incremental_indexing||return false
    db.closed&&return false

    for segment_index = 2:length(db.immutable_segments)
        segment=db.immutable_segments[segment_index]
        segment.index===nothing&&length(segment.vector_store)>0&&return true
    end

    return false
end


function segment_index_build_config(db::VectorDB, count::Int)
    count>0||throw(ArgumentError("cannot index an empty segment"))
    base=db.build_config
    nlists=min(count, base===nothing ? max(1, ceil(Int, sqrt(count))) : base.nlists)
    iterations=base===nothing ? 20 : base.iterations
    seed=base===nothing ? 42 : base.seed
    restarts=base===nothing ? 1 : base.restarts
    return IndexBuildConfig(
        nlists,
        count;
        iterations = iterations,
        seed = seed,
        restarts = restarts,
        training_count = count,
    )
end

function next_segment_index_snapshot(db::VectorDB)
    return with_database_read(db.database_lock) do
        db.closed&&return nothing

        for segment_index = 2:length(db.immutable_segments)
            segment=db.immutable_segments[segment_index]
            segment.index===nothing||continue
            count=length(segment.vector_store)
            count>0||continue
            state=db.segment_index_state
            lock(state.lock)
            hook=try
                state.build_hook
            finally
                unlock(state.lock)
            end
            return (
                id = segment.id,
                vectors = stored_vectors(segment.vector_store),
                metadata = stored_metadata(segment.metadata_store),
                filter_index = segment.filter_index,
                config = segment_index_build_config(db, count),
                metric = db.metric,
                hook = hook,
                max_retries = db.maintenance_config.max_retries,
                retry_delay_ms = db.maintenance_config.retry_delay_ms,
            )
        end

        return nothing
    end
end

function build_segment_index(snapshot)
    snapshot.hook===nothing||snapshot.hook(snapshot)
    config=snapshot.config
    index=build_filter_aware_ivf(
        snapshot.vectors,
        snapshot.metadata;
        nlists = config.nlists,
        iterations = config.iterations,
        seed = config.seed,
        metric = snapshot.metric,
        restarts = config.restarts,
        training_count = config.training_count,
    )
    filter_index=snapshot.filter_index===nothing ? build_bitset_index(snapshot.metadata) :
                 snapshot.filter_index
    return (
        index = index,
        filter_index = filter_index,
        index_bytes = database_index_bytes(index, filter_index),
    )
end

function install_segment_index!(
    db::VectorDB,
    segment_id::AbstractString,
    built,
    config::IndexBuildConfig,
)
    return with_database_write(db.database_lock) do
        db.closed&&return false
        segment_index=findfirst(segment->segment.id==segment_id, db.immutable_segments)
        segment_index===nothing&&return false
        segment=db.immutable_segments[segment_index]
        segment.index===nothing||return false
        replacement=create_immutable_segment(
            segment.id,
            segment.revision_start,
            segment.revision_end,
            segment.vector_store,
            segment.metadata_store,
            segment.id_store;
            excluded = segment.excluded,
            tombstone_ids = segment.tombstone_ids,
            index = built.index,
            filter_index = built.filter_index,
            build_config = config,
            index_bytes = built.index_bytes,
        )
        db.immutable_segments[segment_index]=replacement
        clear_plan_cache!(db)
        validate_database_fast(db)
        return true
    end
end

function segment_index_stop_requested(state::SegmentIndexState)
    lock(state.lock)

    try
        return state.stop_requested
    finally
        unlock(state.lock)
    end
end

function update_segment_index_state!(
    state::SegmentIndexState;
    status::Union{Nothing,Symbol} = nothing,
    current_segment_id = missing,
    attempts::Union{Nothing,Int} = nothing,
    error_message = missing,
    increment_completed::Bool = false,
)
    lock(state.lock)

    try
        status===nothing||(state.status=status)
        current_segment_id===missing||(state.current_segment_id=current_segment_id)
        attempts===nothing||(state.attempts=attempts)
        error_message===missing||(state.last_error=error_message)
        increment_completed&&(state.completed_count+=1)
    finally
        unlock(state.lock)
    end

    return state
end

function segment_index_worker!(db::VectorDB)
    state=db.segment_index_state

    try
        while !segment_index_stop_requested(state)
            snapshot=next_segment_index_snapshot(db)
            snapshot===nothing&&return nothing
            update_segment_index_state!(
                state;
                status = :building,
                current_segment_id = snapshot.id,
                error_message = nothing,
            )
            installed=false

            for attempt = 1:(snapshot.max_retries+1)
                segment_index_stop_requested(state)&&return nothing
                update_segment_index_state!(state; attempts = attempt)

                try
                    built=build_segment_index(snapshot)
                    segment_index_stop_requested(state)&&return nothing
                    installed=install_segment_index!(
                        db,
                        snapshot.id,
                        built,
                        snapshot.config,
                    )
                    installed&&update_segment_index_state!(
                        state;
                        status = :completed,
                        current_segment_id = nothing,
                        error_message = nothing,
                        increment_completed = true,
                    )
                    break
                catch error
                    error isa InterruptException&&rethrow()
                    final_attempt=attempt>snapshot.max_retries
                    update_segment_index_state!(
                        state;
                        status = final_attempt ? :failed : :retrying,
                        error_message = sprint(showerror, error),
                    )
                    final_attempt&&return nothing
                    snapshot.retry_delay_ms>0&&sleep(snapshot.retry_delay_ms/1_000)
                end
            end

            installed||yield()
        end
    finally
        pending=false
        stopped=false
        lock(state.lock)

        try
            pending=state.pending
            stopped=state.stop_requested
            state.pending=false
            state.task=nothing
            stopped&&(state.status=:stopped)
            state.current_segment_id=nothing
        finally
            unlock(state.lock)
        end

        pending&&!stopped&&maybe_schedule_segment_indexing!(db)
    end

    return nothing
end

function schedule_segment_indexing_locked!(db::VectorDB)
    segment_indexing_due_locked(db)||return false
    state=db.segment_index_state
    lock(state.lock)

    try
        state.stop_requested&&return false

        if state.task!==nothing
            state.pending=true
            return false
        end

        state.status=:scheduled
        state.pending=false
        state.last_error=nothing
        state.task=Threads.@spawn segment_index_worker!(db)
        return true
    finally
        unlock(state.lock)
    end
end

function maybe_schedule_segment_indexing!(db::VectorDB)
    return with_database_read(db.database_lock) do
        schedule_segment_indexing_locked!(db)
    end
end

function segment_indexing_status(db::VectorDB)
    due=with_database_read(db.database_lock) do
        segment_indexing_due_locked(db)
    end
    state=db.segment_index_state
    lock(state.lock)

    try
        return (
            due = due,
            status = state.status,
            running = state.task!==nothing,
            pending = state.pending,
            current_segment_id = state.current_segment_id,
            completed_count = state.completed_count,
            attempts = state.attempts,
            last_error = state.last_error,
        )
    finally
        unlock(state.lock)
    end
end

function wait_for_segment_indexing(db::VectorDB; timeout::Real = 60.0)
    timeout>0||throw(ArgumentError("timeout must be positive"))
    deadline=time()+timeout

    while true
        status=segment_indexing_status(db)

        if !status.running
            if status.due&&status.status!=:failed&&status.status!=:stopped
                maybe_schedule_segment_indexing!(db)
            else
                return status
            end
        end

        time()<deadline||throw(ErrorException("segment indexing wait timed out"))
        sleep(0.01)
    end
end

function set_segment_index_build_hook!(db::VectorDB, hook::Union{Nothing,Function})
    state=db.segment_index_state
    lock(state.lock)

    try
        state.build_hook=hook
    finally
        unlock(state.lock)
    end

    return db
end

function set_segment_index_build_hook!(hook::Function, db::VectorDB)
    return set_segment_index_build_hook!(db, hook)
end

function stop_segment_indexing!(db::VectorDB)
    state=db.segment_index_state
    lock(state.lock)
    task=nothing

    try
        state.stop_requested=true
        state.pending=false
        task=state.task
    finally
        unlock(state.lock)
    end

    task===nothing||task===current_task()||wait(task)
    return db
end

function configure_maintenance!(db::VectorDB, config::MaintenanceConfig)
    with_database_write(db.database_lock) do
        ensure_database_open(db)
        previous_config=db.maintenance_config
        db.maintenance_config=config

        try
            ensure_delta_search_capacity!(db, Any[])
            maybe_seal_active_segment_locked!(db)
        catch
            db.maintenance_config=previous_config
            rethrow()
        end

        state=db.maintenance_state
        lock(state.lock)

        try
            state.stop_requested=false
            config.enabled||(state.pending=false)
        finally
            unlock(state.lock)
        end

        segment_state=db.segment_index_state
        lock(segment_state.lock)

        try
            segment_state.stop_requested=!config.incremental_indexing
            config.incremental_indexing||(segment_state.pending=false)
        finally
            unlock(segment_state.lock)
        end
    end

    maybe_schedule_database_maintenance!(db)
    config.incremental_indexing ? maybe_schedule_segment_indexing!(db) :
    stop_segment_indexing!(db)
    return db
end

function maintenance_status(db::VectorDB)
    database=with_database_read(db.database_lock) do
        counts=maintenance_counts_locked(db)
        return merge(
            counts,
            (
                revision = db.revision,
                index_revision = db.index_revision,
                enabled = db.maintenance_config.enabled,
                due = database_maintenance_due_locked(db),
            ),
        )
    end
    state=db.maintenance_state
    lock(state.lock)

    try
        return merge(
            database,
            (
                status = state.status,
                running = state.task!==nothing,
                pending = state.pending,
                attempts = state.attempts,
                last_started_revision = state.last_started_revision,
                last_completed_revision = state.last_completed_revision,
                last_duration_ms = state.last_duration_ms,
                last_error = state.last_error,
            ),
        )
    finally
        unlock(state.lock)
    end
end

function wait_for_maintenance(db::VectorDB; timeout::Real = 60.0)
    timeout>0||throw(ArgumentError("timeout must be positive"))
    deadline=time()+timeout

    while true
        status=maintenance_status(db)
        if !status.running
            if status.due&&status.status!=:failed&&status.status!=:stopped
                maybe_schedule_database_maintenance!(db)
            else
                return status
            end
        end

        time()<deadline||throw(ErrorException("maintenance wait timed out"))
        sleep(0.01)
    end
end

function stop_database_maintenance!(db::VectorDB)
    state=db.maintenance_state
    lock(state.lock)
    task=nothing

    try
        state.stop_requested=true
        state.pending=false
        task=state.task
    finally
        unlock(state.lock)
    end

    task===nothing||task===current_task()||wait(task)
    return db
end
