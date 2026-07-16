const DATABASE_WRITER_LOCK_FILE="writer.lock"
const DATABASE_LOCK_EXCLUSIVE=Cint(2)
const DATABASE_LOCK_NONBLOCKING=Cint(4)
const DATABASE_LOCK_UNLOCK=Cint(8)

function database_writer_lock_path(path::AbstractString)
    return joinpath(path, DATABASE_WRITER_LOCK_FILE)
end

function sync_writer_lock_io(io::IO)
    flush(io)
    ccall(:fsync, Cint, (Cint,), fd(io))==0||error("failed to sync database writer lock")
    return io
end

function acquire_database_writer_lock(path::AbstractString)
    Sys.isunix()||throw(
        ArgumentError("durable writer ownership is currently supported on Unix systems"),
    )
    mkpath(path)
    lock_path=database_writer_lock_path(path)
    io=open(lock_path, "a+")
    result=ccall(
        :flock,
        Cint,
        (Cint, Cint),
        fd(io),
        DATABASE_LOCK_EXCLUSIVE|DATABASE_LOCK_NONBLOCKING,
    )

    if result!=0
        seekstart(io)
        owner=strip(read(io, String))
        close(io)
        detail=isempty(owner) ? "another process or handle" : owner
        throw(ArgumentError("database already has a writable owner ($(detail))"))
    end

    try
        seekstart(io)
        truncate(io, 0)
        write(io, "pid=$(getpid())\n")
        sync_writer_lock_io(io)
    catch
        ccall(:flock, Cint, (Cint, Cint), fd(io), DATABASE_LOCK_UNLOCK)
        close(io)
        rethrow()
    end

    return io
end

function release_database_writer_lock(io::IOStream)
    isopen(io)||return nothing
    ccall(:flock, Cint, (Cint, Cint), fd(io), DATABASE_LOCK_UNLOCK)
    close(io)
    return nothing
end

function release_database_writer_lock_finalizer!(db::VectorDB)
    io=db.writer_lock
    db.writer_lock=nothing
    db.closed=true
    release_vector_store_mapping!(db.vector_store)

    if io!==nothing
        try
            release_database_writer_lock(io)
        catch
        end
    end

    return nothing
end

function attach_database_writer_lock!(db::VectorDB, io::IOStream)
    db.closed&&throw(ArgumentError("database handle is closed"))
    db.writer_lock===nothing||throw(
        ArgumentError("database handle already owns a writer lock"),
    )
    db.writer_lock=io
    finalizer(release_database_writer_lock_finalizer!, db)
    return db
end

function ensure_database_open(db::VectorDB)
    db.closed&&throw(ArgumentError("database handle is closed"))
    return db
end

function ensure_database_writer_ownership(db::VectorDB)
    ensure_database_open(db)
    db.writer_lock===nothing&&throw(
        ArgumentError("database handle does not own the durable writer lock"),
    )
    isopen(db.writer_lock)||throw(ArgumentError("database writer lock is closed"))
    return db
end

function activate_database_writer_lock!(db::VectorDB)
    ensure_database_open(db)
    db.writer_lock===nothing||return db
    io=acquire_database_writer_lock(db.path)

    try
        wal_path=database_wal_path(db.path)
        current_path=database_current_path(db.path)
        manifest_path=joinpath(db.path, "manifest.toml")
        any(isfile, (wal_path, current_path, manifest_path))&&throw(
            ArgumentError("database already exists; load it instead"),
        )
        attach_database_writer_lock!(db, io)
    catch
        release_database_writer_lock(io)
        rethrow()
    end

    return db
end

function Base.close(db::VectorDB)
    stop_database_maintenance!(db)
    stop_segment_indexing!(db)

    return with_database_write(db.database_lock) do
        db.closed&&return nothing
        io=db.writer_lock
        db.writer_lock=nothing
        db.closed=true
        released=IdSet{VectorStore}()

        for segment in db.immutable_segments
            segment.vector_store in released&&continue
            push!(released, segment.vector_store)
            release_vector_store_mapping!(segment.vector_store)
        end

        if !(db.vector_store in released)
            release_vector_store_mapping!(db.vector_store)
        end
        io===nothing||release_database_writer_lock(io)
        return nothing
    end
end

function Base.isopen(db::VectorDB)
    return !db.closed
end
