const DATABASE_STORAGE_FAULT_KEY=:sen_database_storage_fault
const DATABASE_STORAGE_FAULT_STAGES=(
    :wal_append,
    :snapshot_commit,
    :current_pointer,
    :wal_checkpoint,
)
const DATABASE_PROCESS_CRASH_STAGES=(
    :before_wal_append,
    :after_wal_sync,
    :after_snapshot_seal,
    :after_snapshot_commit,
)
const DATABASE_CRASH_STAGE_ENV="SEN_CRASH_STAGE"
const DATABASE_CRASH_MARKER_ENV="SEN_CRASH_MARKER"

struct InjectedDatabaseStorageError <: Exception
    stage::Symbol
end

function Base.showerror(io::IO, error::InjectedDatabaseStorageError)
    print(
        io,
        "injected database storage failure at ",
        error.stage,
        ": no space left on device (ENOSPC)",
    )
end

function inject_database_storage_fault!(stage::Symbol)
    stage in DATABASE_STORAGE_FAULT_STAGES||throw(
        ArgumentError("unsupported database storage fault stage"),
    )
    task_local_storage()[DATABASE_STORAGE_FAULT_KEY]=stage
    return stage
end

function clear_database_storage_fault!()
    storage=task_local_storage()
    haskey(storage, DATABASE_STORAGE_FAULT_KEY)&&delete!(
        storage,
        DATABASE_STORAGE_FAULT_KEY,
    )
    return nothing
end

function maybe_inject_database_storage_fault!(stage::Symbol)
    storage=task_local_storage()
    get(storage, DATABASE_STORAGE_FAULT_KEY, nothing)===stage||return nothing
    delete!(storage, DATABASE_STORAGE_FAULT_KEY)
    throw(InjectedDatabaseStorageError(stage))
end

function maybe_pause_database_process!(stage::Symbol, path::AbstractString)
    stage in DATABASE_PROCESS_CRASH_STAGES||throw(
        ArgumentError("unsupported database crash stage"),
    )
    get(ENV, DATABASE_CRASH_STAGE_ENV, "")==String(stage)||return nothing
    marker=get(ENV, DATABASE_CRASH_MARKER_ENV, "")
    isempty(marker)&&error("database crash marker is required")
    mkpath(dirname(marker))

    open(marker, "w") do io
        println(io, "stage=", stage)
        println(io, "pid=", getpid())
        println(io, "path=", abspath(path))
        flush(io)
        ccall(:fsync, Cint, (Cint,), fd(io))==0||error(
            "failed to sync database crash marker",
        )
    end

    fsync_path(dirname(marker))
    wait(Base.Event())
    return nothing
end
