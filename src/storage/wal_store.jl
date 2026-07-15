using SHA

const DATABASE_WAL_FILE="wal.bin"
const DATABASE_WAL_MAGIC=UInt8[0x53,0x45,0x4e,0x57,0x41,0x4c,0x30,0x31]
const DATABASE_WAL_FORMAT_VERSION=1
const DATABASE_WAL_PUT=UInt8(1)
const DATABASE_WAL_DELETE=UInt8(2)
const DATABASE_WAL_CHECKSUM_BYTES=32
const DATABASE_WAL_HEADER_BODY_BYTES=27
const DATABASE_WAL_MAX_RECORD_BYTES=1_073_741_824

struct DatabaseWALRecord
    revision::UInt64
    operation::UInt8
    ids::Vector{Any}
    vectors::Vector{Vector{Float32}}
    metadata::Vector{NamedTuple}
end

function database_wal_path(path::AbstractString)
    return joinpath(path,DATABASE_WAL_FILE)
end

function database_wal_is_active(db::VectorDB)
    return isfile(database_wal_path(db.path))||isfile(database_current_path(db.path))||isfile(joinpath(db.path,"manifest.toml"))
end

function database_wal_metric_code(metric::Symbol)
    metric===:cosine&&return UInt8(1)
    metric===:dot&&return UInt8(2)
    throw(ArgumentError("unsupported WAL metric"))
end

function database_wal_metric(code::UInt8)
    code==UInt8(1)&&return :cosine
    code==UInt8(2)&&return :dot
    throw(ArgumentError("invalid stored WAL metric"))
end

function sync_wal_io(io::IO)
    flush(io)
    ccall(:fsync,Cint,(Cint,),fd(io))==0||error("failed to sync database WAL")
    return io
end

function database_wal_header_body(revision::UInt64,dim::Int,metric::Symbol)
    dim>0||throw(ArgumentError("WAL dimension must be positive"))
    io=IOBuffer()
    write(io,DATABASE_WAL_MAGIC)
    write_portable_uint16(io,DATABASE_WAL_FORMAT_VERSION)
    write_portable_uint64(io,revision)
    write_portable_int64(io,dim)
    write(io,database_wal_metric_code(metric))
    return take!(io)
end

function write_database_wal_header(io::IO,revision::UInt64,dim::Int,metric::Symbol)
    body=database_wal_header_body(revision,dim,metric)
    length(body)==DATABASE_WAL_HEADER_BODY_BYTES||error("database WAL header size is invalid")
    write(io,body)
    write(io,sha256(body))
    return io
end

function read_database_wal_header(io::IO)
    try
        body=read(io,DATABASE_WAL_HEADER_BODY_BYTES)
        length(body)==DATABASE_WAL_HEADER_BODY_BYTES||throw(ArgumentError("database WAL header is truncated"))
        checksum=read(io,DATABASE_WAL_CHECKSUM_BYTES)
        length(checksum)==DATABASE_WAL_CHECKSUM_BYTES||throw(ArgumentError("database WAL header is truncated"))
        sha256(body)==checksum||throw(ArgumentError("database WAL header checksum does not match"))
        header=IOBuffer(body)
        magic=read(header,length(DATABASE_WAL_MAGIC))
        magic==DATABASE_WAL_MAGIC||throw(ArgumentError("invalid database WAL"))
        version=Int(read_portable_uint16(header))
        version==DATABASE_WAL_FORMAT_VERSION||throw(ArgumentError("unsupported database WAL format version"))
        revision=read_portable_uint64(header)
        dim=Int(read_portable_int64(header))
        dim>0||throw(ArgumentError("stored WAL dimension must be positive"))
        metric=database_wal_metric(read(header,UInt8))
        eof(header)||throw(ArgumentError("database WAL header contains unexpected data"))
        return(revision=revision,dim=dim,metric=metric,header_bytes=position(io),)
    catch error
        portable_read_error(error,"database WAL header")
    end
end

function replace_database_wal(path::AbstractString,revision::UInt64,dim::Int,metric::Symbol)
    mkpath(path)
    wal_path=database_wal_path(path)
    temporary_path=joinpath(path,".wal-$(time_ns())-$(getpid())")

    try
        open(temporary_path,"w") do io
            write_database_wal_header(io,revision,dim,metric)
            sync_wal_io(io)
        end

        Base.rename(temporary_path,wal_path)
        fsync_path(path)
    catch
        isfile(temporary_path)&&rm(temporary_path;force=true,)
        rethrow()
    end

    return wal_path
end


function ensure_database_wal(db::VectorDB)
    wal_path=database_wal_path(db.path)

    if !isfile(wal_path)
        replace_database_wal(db.path,db.revision,db.dim,db.metric)
        db.wal_revision=db.revision
        return wal_path
    end

    if db.wal_revision!==nothing
        db.wal_revision==db.revision||throw(ArgumentError("database WAL is ahead of memory; reload the database"))
        return wal_path
    end

    wal=read_database_wal(db.path;repair_tail=true,)
    header=wal.header

    header.dim==db.dim||throw(DimensionMismatch("WAL dimension doesnt match database"))
    header.metric===db.metric||throw(ArgumentError("WAL metric doesnt match database"))
    last_revision=isempty(wal.records) ? header.revision : last(wal.records).revision
    db.wal_revision=last_revision
    last_revision==db.revision||throw(ArgumentError("database WAL is ahead of memory; reload the database"))

    return wal_path
end

function write_database_wal_put_body(io::IO,revision::UInt64,vectors,metadata,ids)
    count=length(ids)
    length(vectors)==count||throw(DimensionMismatch("WAL vector count doesnt match id count"))
    length(metadata)==count||throw(DimensionMismatch("WAL metadata count doesnt match id count"))
    write_portable_uint64(io,revision)
    write(io,DATABASE_WAL_PUT)
    write_portable_length(io,count,"WAL record count")

    for index in 1:count
        write_portable_value(io,ids[index])
        write_portable_value(io,vectors[index])
        write_portable_named_tuple(io,metadata[index])
    end

    return io
end

function write_database_wal_delete_body(io::IO,revision::UInt64,ids)
    write_portable_uint64(io,revision)
    write(io,DATABASE_WAL_DELETE)
    write_portable_length(io,length(ids),"WAL record count")

    for id in ids
        write_portable_value(io,id)
    end

    return io
end

function append_database_wal_body!(db::VectorDB,revision::UInt64,body::Vector{UInt8})
    length(body)<=DATABASE_WAL_MAX_RECORD_BYTES||throw(ArgumentError("database WAL record is too large"))
    wal_path=ensure_database_wal(db)
    revision==next_database_revision(db)||throw(ArgumentError("database WAL append revision is invalid"))
    checksum=sha256(body)

    try
        open(wal_path,"a") do io
            write_portable_uint64(io,length(body))
            write(io,body)
            write(io,checksum)
            sync_wal_io(io)
        end

        db.wal_revision=revision
    catch
        db.wal_revision=nothing
        rethrow()
    end

    return wal_path
end

function append_database_wal_put!(db::VectorDB,revision::UInt64,vectors,metadata,ids)
    database_wal_is_active(db)||return nothing
    io=IOBuffer()
    write_database_wal_put_body(io,revision,vectors,metadata,ids)
    return append_database_wal_body!(db,revision,take!(io))
end

function append_database_wal_delete!(db::VectorDB,revision::UInt64,ids)
    database_wal_is_active(db)||return nothing
    io=IOBuffer()
    write_database_wal_delete_body(io,revision,ids)
    return append_database_wal_body!(db,revision,take!(io))
end

function read_database_wal_record(body::Vector{UInt8},dim::Int)
    io=IOBuffer(body)

    try
        revision=read_portable_uint64(io)
        operation=read(io,UInt8)
        count=read_portable_length(io,"WAL record count")
        ids=Vector{Any}(undef,count)
        vectors=Vector{Vector{Float32}}()
        metadata=Vector{NamedTuple}()

        if operation==DATABASE_WAL_PUT
            sizehint!(vectors,count)
            sizehint!(metadata,count)

            for index in 1:count
                id=read_portable_value(io)
                id===nothing&&throw(ArgumentError("stored WAL id cannot be nothing"))
                vector=read_portable_value(io)
                vector isa Vector{Float32}||throw(ArgumentError("stored WAL vector is invalid"))
                length(vector)==dim||throw(DimensionMismatch("stored WAL vector dimension doesnt match database"))
                ids[index]=id
                push!(vectors,vector)
                push!(metadata,read_portable_named_tuple(io))
            end
        elseif operation==DATABASE_WAL_DELETE
            for index in 1:count
                id=read_portable_value(io)
                id===nothing&&throw(ArgumentError("stored WAL id cannot be nothing"))
                ids[index]=id
            end
        else
            throw(ArgumentError("unsupported database WAL operation"))
        end

        eof(io)||throw(ArgumentError("database WAL record contains unexpected data"))
        return DatabaseWALRecord(revision,operation,ids,vectors,metadata)
    catch error
        portable_read_error(error,"database WAL record")
    end
end

function read_database_wal(path::AbstractString;repair_tail::Bool=false,)
    wal_path=database_wal_path(path)
    isfile(wal_path)||return nothing
    records=DatabaseWALRecord[]
    valid_bytes=0
    incomplete_tail=false

    header=open(wal_path,"r") do io
        header=read_database_wal_header(io)
        valid_bytes=header.header_bytes
        expected_revision=header.revision
        file_size=filesize(wal_path)

        while position(io)<file_size
            remaining=file_size-position(io)

            if remaining<sizeof(UInt64)
                incomplete_tail=true
                break
            end

            body_length=read_portable_uint64(io)
            body_length<=UInt64(DATABASE_WAL_MAX_RECORD_BYTES)||throw(ArgumentError("stored database WAL record is too large"))
            required=body_length+UInt64(DATABASE_WAL_CHECKSUM_BYTES)
            remaining=UInt64(file_size-position(io))

            if remaining<required
                incomplete_tail=true
                break
            end

            body=read(io,Int(body_length))
            checksum=read(io,DATABASE_WAL_CHECKSUM_BYTES)
            sha256(body)==checksum||throw(ArgumentError("database WAL checksum does not match"))
            record=read_database_wal_record(body,header.dim)
            expected_revision==typemax(UInt64)&&throw(ArgumentError("database WAL revision overflow"))
            expected_revision+=UInt64(1)
            record.revision==expected_revision||throw(ArgumentError("database WAL revisions are not contiguous"))
            push!(records,record)
            valid_bytes=position(io)
        end

        return header
    end

    if incomplete_tail&&repair_tail
        open(wal_path,"r+") do io
            truncate(io,valid_bytes)
            sync_wal_io(io)
        end
    end

    return(header=header,records=records,incomplete_tail=incomplete_tail,valid_bytes=valid_bytes,)
end

function apply_database_wal_record!(db::VectorDB,record::DatabaseWALRecord)
    record.revision==db.revision+UInt64(1)||throw(ArgumentError("database WAL revision doesnt follow snapshot"))

    if record.operation==DATABASE_WAL_PUT
        length(Set{Any}(record.ids))==length(record.ids)||throw(ArgumentError("database WAL put ids must be unique"))

        for index in eachindex(record.ids)
            id=record.ids[index]
            existing=has_database_id(db,id)
            update_database_record!(db,id,record.vectors[index],record.metadata[index])
            existing||(db.live_count+=1)
        end
    elseif record.operation==DATABASE_WAL_DELETE
        length(Set{Any}(record.ids))==length(record.ids)||throw(ArgumentError("database WAL delete ids must be unique"))

        for id in record.ids
            has_database_id(db,id)||throw(ArgumentError("database WAL deletes a missing id"))
            delete_database_record!(db,id)
            db.live_count-=1
        end
    else
        throw(ArgumentError("unsupported database WAL operation"))
    end

    db.revision=record.revision
    clear_plan_cache!(db)
    validate_database_fast(db)
    return db
end

function replay_database_wal!(db::VectorDB)
    wal=read_database_wal(db.path;repair_tail=true,)
    wal===nothing&&return db
    wal.header.dim==db.dim||throw(DimensionMismatch("WAL dimension doesnt match database"))
    wal.header.metric===db.metric||throw(ArgumentError("WAL metric doesnt match database"))
    last_revision=isempty(wal.records) ? wal.header.revision : last(wal.records).revision
    db.wal_revision=last_revision
    wal.header.revision>db.revision&&return db

    for record in wal.records
        record.revision<=db.revision&&continue
        apply_database_wal_record!(db,record)
    end

    return db
end

function checkpoint_database_wal!(db::VectorDB)
    try
        path=replace_database_wal(db.path,db.revision,db.dim,db.metric)
        db.wal_revision=db.revision
        return path
    catch
        db.wal_revision=nothing
        rethrow()
    end
end
