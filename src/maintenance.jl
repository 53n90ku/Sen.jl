function maintenance_counts_locked(db::VectorDB)
    base_count=length(db.vector_store)
    delta_count=length(db.delta_store)
    tombstone_count=count(db.base_tombstones)
    denominator=max(1,base_count)
    return(
        base_count=base_count,
        delta_count=delta_count,
        tombstone_count=tombstone_count,
        delta_ratio=delta_count/denominator,
        tombstone_ratio=tombstone_count/denominator,
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
    counts.delta_count+counts.tombstone_count>=config.minimum_changes||return false
    delta_due=(config.delta_threshold>0&&counts.delta_count>=config.delta_threshold)||(config.delta_ratio>0&&counts.delta_ratio>=config.delta_ratio)
    tombstone_due=(config.tombstone_threshold>0&&counts.tombstone_count>=config.tombstone_threshold)||(config.tombstone_ratio>0&&counts.tombstone_ratio>=config.tombstone_ratio)
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

function update_maintenance_state!(db::VectorDB;status::Union{Nothing,Symbol}=nothing,attempts::Union{Nothing,Int}=nothing,started_revision=nothing,completed_revision=nothing,duration_ms::Union{Nothing,Float64}=nothing,error_message=nothing,clear_error::Bool=false,)
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
        save!(db;retain_snapshots=db.checkpoint_retain_snapshots,)
        return nothing
    catch error
        error isa InterruptException&&rethrow()
        message=sprint(showerror,error)
        update_maintenance_state!(db;error_message="index rebuilt but automatic snapshot failed: $(message)",)
        @warn "automatic maintenance snapshot failed; rebuilt index remains available and WAL remains durable" exception=(error,catch_backtrace())
        return nothing
    end
end

function rebuild_database_maintenance!(db::VectorDB)
    build=with_database_read(db.database_lock) do
        config=db.build_config
        config===nothing&&throw(ArgumentError("database has no previous index build configuration"))
        count=logical_length(db)
        count>0||throw(ArgumentError("database cannot be empty"))
        nlists=min(config.nlists,count)
        training_count=min(count,max(nlists,config.training_count))
        return(nlists=nlists,iterations=config.iterations,seed=config.seed,restarts=config.restarts,training_count=training_count,)
    end

    return build!(db;build...)
end

function database_maintenance_worker!(db::VectorDB)
    state=db.maintenance_state
    config=db.maintenance_config
    started_ns=time_ns()

    try
        for attempt in 1:(config.max_retries+1)
            if maintenance_stop_requested(db)
                update_maintenance_state!(db;status=:stopped,duration_ms=(time_ns()-started_ns)/1_000_000,)
                return nothing
            end

            if !database_maintenance_due(db)
                update_maintenance_state!(db;status=:idle,duration_ms=(time_ns()-started_ns)/1_000_000,)
                return nothing
            end

            revision=with_database_read(db.database_lock) do
                db.revision
            end
            update_maintenance_state!(db;status=:building,attempts=attempt,started_revision=revision,)

            try
                rebuild_database_maintenance!(db)
                completed_revision=with_database_read(db.database_lock) do
                    db.index_revision
                end
                update_maintenance_state!(db;status=:completed,completed_revision=completed_revision,duration_ms=(time_ns()-started_ns)/1_000_000,clear_error=true,)
                persist_maintenance_rebuild!(db)
                return nothing
            catch error
                error isa InterruptException&&rethrow()
                message=sprint(showerror,error)
                final_attempt=attempt>config.max_retries
                update_maintenance_state!(db;status=final_attempt ? :failed : :retrying,duration_ms=(time_ns()-started_ns)/1_000_000,error_message=message,)

                if final_attempt
                    @warn "automatic database maintenance failed; current base and delta remain searchable" exception=(error,catch_backtrace())
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

function configure_maintenance!(db::VectorDB,config::MaintenanceConfig)
    with_database_write(db.database_lock) do
        ensure_database_open(db)
        db.maintenance_config=config
        state=db.maintenance_state
        lock(state.lock)

        try
            state.stop_requested=false
            config.enabled||(state.pending=false)
        finally
            unlock(state.lock)
        end
    end

    maybe_schedule_database_maintenance!(db)
    return db
end

function maintenance_status(db::VectorDB)
    database=with_database_read(db.database_lock) do
        counts=maintenance_counts_locked(db)
        return merge(counts,(
            revision=db.revision,
            index_revision=db.index_revision,
            enabled=db.maintenance_config.enabled,
            due=database_maintenance_due_locked(db),
        ))
    end
    state=db.maintenance_state
    lock(state.lock)

    try
        return merge(database,(
            status=state.status,
            running=state.task!==nothing,
            pending=state.pending,
            attempts=state.attempts,
            last_started_revision=state.last_started_revision,
            last_completed_revision=state.last_completed_revision,
            last_duration_ms=state.last_duration_ms,
            last_error=state.last_error,
        ))
    finally
        unlock(state.lock)
    end
end

function wait_for_maintenance(db::VectorDB;timeout::Real=60.0,)
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
