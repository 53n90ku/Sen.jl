mutable struct DatabaseLock
    condition::Threads.Condition
    readers::Int
    reader_depth::IdDict{Task,Int}
    writer::Union{Nothing,Task}
    writer_depth::Int
    waiting_writers::Int
end

function DatabaseLock()
    gate=ReentrantLock()
    return DatabaseLock(Threads.Condition(gate), 0, IdDict{Task,Int}(), nothing, 0, 0)
end

function lock_database_read!(lock::DatabaseLock)
    task=current_task()
    Base.lock(lock.condition)

    try
        nested=get(lock.reader_depth, task, 0)>0||lock.writer===task

        while !nested&&(lock.writer!==nothing||lock.waiting_writers>0)
            wait(lock.condition)
        end

        lock.readers+=1
        lock.reader_depth[task]=get(lock.reader_depth, task, 0)+1
    finally
        Base.unlock(lock.condition)
    end

    return lock
end

function unlock_database_read!(lock::DatabaseLock)
    task=current_task()
    Base.lock(lock.condition)

    try
        depth=get(lock.reader_depth, task, 0)
        depth>0||throw(ArgumentError("database read lock is not held by this task"))

        if depth==1
            delete!(lock.reader_depth, task)
        else
            lock.reader_depth[task]=depth-1
        end

        lock.readers-=1
        lock.readers==0&&notify(lock.condition; all = true)
    finally
        Base.unlock(lock.condition)
    end

    return lock
end

function lock_database_write!(lock::DatabaseLock)
    task=current_task()
    Base.lock(lock.condition)

    try
        if lock.writer===task
            lock.writer_depth+=1
            return lock
        end

        get(lock.reader_depth, task, 0)==0||throw(
            ArgumentError("database lock cannot be upgraded from read to write"),
        )
        lock.waiting_writers+=1

        try
            while lock.writer!==nothing||lock.readers>0
                wait(lock.condition)
            end

            lock.writer=task
            lock.writer_depth=1
        finally
            lock.waiting_writers-=1
        end
    finally
        Base.unlock(lock.condition)
    end

    return lock
end

function unlock_database_write!(lock::DatabaseLock)
    task=current_task()
    Base.lock(lock.condition)

    try
        lock.writer===task||throw(
            ArgumentError("database write lock is not held by this task"),
        )
        lock.writer_depth-=1

        if lock.writer_depth==0
            lock.writer=nothing
            notify(lock.condition; all = true)
        end
    finally
        Base.unlock(lock.condition)
    end

    return lock
end

function with_database_read(f::Function, lock::DatabaseLock)
    lock_database_read!(lock)

    try
        return f()
    finally
        unlock_database_read!(lock)
    end
end

function with_database_write(f::Function, lock::DatabaseLock)
    lock_database_write!(lock)

    try
        return f()
    finally
        unlock_database_write!(lock)
    end
end
