using SHA

const DATABASE_SNAPSHOT_DIRECTORY="snapshots"
const DATABASE_CURRENT_FILE="CURRENT"
const DATABASE_SNAPSHOT_DESCRIPTOR="snapshot.toml"
const DATABASE_SNAPSHOT_FORMAT_VERSION=1

function database_snapshot_root(path::AbstractString)
    return joinpath(path,DATABASE_SNAPSHOT_DIRECTORY)
end

function database_current_path(path::AbstractString)
    return joinpath(path,DATABASE_CURRENT_FILE)
end

function snapshot_descriptor_path(path::AbstractString)
    return joinpath(path,DATABASE_SNAPSHOT_DESCRIPTOR)
end

function current_database_generation(path::AbstractString)
    current_path=database_current_path(path)
    isfile(current_path)||return nothing
    return validate_snapshot_generation(strip(read(current_path,String)))
end

function new_database_snapshot(path::AbstractString)
    mkpath(path)
    root=database_snapshot_root(path)
    mkpath(root)
    generation="$(time_ns())-$(getpid())"
    temporary_path=joinpath(root,".tmp-$(generation)")
    mkpath(temporary_path)

    return(generation=generation,temporary_path=temporary_path,)
end

function validate_snapshot_generation(generation::AbstractString)
    isempty(generation)&&throw(ArgumentError("database snapshot generation cannot be empty"))
    basename(generation)==generation||throw(ArgumentError("invalid database snapshot generation"))
    startswith(generation,".")&&throw(ArgumentError("invalid database snapshot generation"))
    return String(generation)
end

function snapshot_data_files(path::AbstractString)
    files=String[]

    for entry in readdir(path)
        entry==DATABASE_SNAPSHOT_DESCRIPTOR&&continue
        startswith(entry,".")&&continue
        isfile(joinpath(path,entry))||continue
        push!(files,entry)
    end

    sort!(files)
    return files
end

function snapshot_file_hash(path::AbstractString)
    return open(path,"r") do io
        bytes2hex(sha256(io))
    end
end

function fsync_path(path::AbstractString)
    descriptor=ccall(:open,Cint,(Cstring,Cint),path,0)
    descriptor>=0||error("failed to open path for syncing")

    try
        ccall(:fsync,Cint,(Cint,),descriptor)==0||error("failed to sync path")
    finally
        ccall(:close,Cint,(Cint,),descriptor)
    end

    return path
end

function seal_database_snapshot(path::AbstractString,revision::UInt64)
    files=snapshot_data_files(path)
    required=Set(["manifest.toml","vectors.bin","metadata.bin","ids.bin"])
    required⊆Set(files)||throw(ArgumentError("database snapshot is missing required files"))
    file_data=Dict{String,Any}()

    for file in files
        file_path=joinpath(path,file)
        file_data[file]=Dict(
            "size"=>filesize(file_path),
            "sha256"=>snapshot_file_hash(file_path),
        )
    end

    descriptor=Dict(
        "format_version"=>DATABASE_SNAPSHOT_FORMAT_VERSION,
        "revision"=>Int(revision),
        "files"=>file_data,
    )
    descriptor_path=snapshot_descriptor_path(path)

    open(descriptor_path,"w") do io
        TOML.print(io,descriptor)
    end

    for file in files
        fsync_path(joinpath(path,file))
    end

    fsync_path(descriptor_path)
    fsync_path(path)
    return descriptor_path
end

function validate_database_snapshot(path::AbstractString;allow_legacy::Bool=true,)
    isdir(path)||throw(ArgumentError("database snapshot does not exist"))
    descriptor_path=snapshot_descriptor_path(path)

    if !isfile(descriptor_path)
        allow_legacy||throw(ArgumentError("database snapshot is not sealed"))

        for file in ("manifest.toml","vectors.bin","metadata.bin","ids.bin")
            isfile(joinpath(path,file))||throw(ArgumentError("database snapshot is missing required files"))
        end

        return(revision=nothing,legacy=true,)
    end

    descriptor=TOML.parsefile(descriptor_path)

    for key in ("format_version","revision","files")
        haskey(descriptor,key)||throw(ArgumentError("snapshot descriptor is missing required fields"))
    end

    Int(descriptor["format_version"])==DATABASE_SNAPSHOT_FORMAT_VERSION||throw(ArgumentError("unsupported snapshot format version"))
    revision=Int(descriptor["revision"])
    revision>=0||throw(ArgumentError("snapshot revision cannot be negative"))
    stored_files=descriptor["files"]
    stored_files isa AbstractDict||throw(ArgumentError("snapshot file table is invalid"))
    actual_files=snapshot_data_files(path)
    Set(String.(keys(stored_files)))==Set(actual_files)||throw(ArgumentError("snapshot files do not match descriptor"))

    for file in actual_files
        entry=stored_files[file]
        entry isa AbstractDict||throw(ArgumentError("snapshot file entry is invalid"))
        haskey(entry,"size")&&haskey(entry,"sha256")||throw(ArgumentError("snapshot file entry is incomplete"))
        file_path=joinpath(path,file)
        filesize(file_path)==Int(entry["size"])||throw(ArgumentError("snapshot file size does not match descriptor"))
        snapshot_file_hash(file_path)==String(entry["sha256"])||throw(ArgumentError("snapshot file checksum does not match descriptor"))
    end

    return(revision=UInt64(revision),legacy=false,)
end

function write_database_current(path::AbstractString,generation::AbstractString)
    generation=validate_snapshot_generation(generation)
    pointer_temporary=joinpath(path,".CURRENT-$(generation)-$(time_ns())")

    open(pointer_temporary,"w") do io
        write(io,generation)
        write(io,'\n')
        flush(io)
        ccall(:fsync,Cint,(Cint,),fd(io))==0||error("failed to sync database current pointer")
    end

    Base.rename(pointer_temporary,database_current_path(path))
    fsync_path(path)
    return database_current_path(path)
end

function commit_database_snapshot(path::AbstractString,snapshot)
    validate_database_snapshot(snapshot.temporary_path;allow_legacy=false,)
    generation=validate_snapshot_generation(snapshot.generation)
    final_path=joinpath(database_snapshot_root(path),generation)
    ispath(final_path)&&throw(ArgumentError("database snapshot already exists"))
    mv(snapshot.temporary_path,final_path)
    fsync_path(database_snapshot_root(path))
    write_database_current(path,generation)
    return final_path
end

function abort_database_snapshot(snapshot)
    isdir(snapshot.temporary_path)&&rm(snapshot.temporary_path;recursive=true,force=true,)
    return nothing
end

function current_database_snapshot(path::AbstractString)
    generation=current_database_generation(path)
    generation===nothing&&return String(path)
    snapshot_path=joinpath(database_snapshot_root(path),generation)
    isdir(snapshot_path)||throw(ArgumentError("current database snapshot does not exist"))
    return snapshot_path
end

function database_snapshot_generations(path::AbstractString)
    root=database_snapshot_root(path)
    isdir(root)||return String[]
    generations=String[]

    for entry in readdir(root)
        startswith(entry,".")&&continue
        isdir(joinpath(root,entry))||continue
        push!(generations,entry)
    end

    sort!(generations;rev=true,)
    return generations
end

function validate_database_snapshot_contents(path::AbstractString)
    descriptor=validate_database_snapshot(path)
    manifest=load_manifest(path)
    vector_store=load_vector_store(path)
    metadata_store=load_metadata_store(path)
    id_store=load_id_store(path)
    vector_store.dim==manifest.dim||throw(DimensionMismatch("stored vector dimension doesnt match manifest"))
    length(vector_store)==manifest.count||throw(DimensionMismatch("stored vector count doesnt match manifest"))
    length(metadata_store)==manifest.count||throw(DimensionMismatch("stored metadata count doesnt match manifest"))
    length(id_store)==manifest.count||throw(DimensionMismatch("stored id count doesnt match manifest"))
    descriptor.revision===nothing||descriptor.revision==manifest.revision||throw(ArgumentError("snapshot revision doesnt match manifest"))

    if manifest.index_revision==manifest.revision
        isfile(index_file_path(path))||throw(ArgumentError("snapshot is missing its current index"))
        ivf=load_ivf_index(path)
        sum(length,ivf.lists)==manifest.count||throw(DimensionMismatch("stored index count doesnt match manifest"))
    end

    return true
end

function recover_database_snapshot(path::AbstractString;repair::Bool=true,)
    for generation in database_snapshot_generations(path)
        snapshot_path=joinpath(database_snapshot_root(path),generation)

        try
            validate_database_snapshot_contents(snapshot_path)
            repair&&write_database_current(path,generation)
            return snapshot_path
        catch error
            error isa InterruptException&&rethrow()
        end
    end

    throw(ArgumentError("database has no valid snapshot generations"))
end

function prune_database_snapshots(path::AbstractString;retain::Int=2,)
    retain>0||throw(ArgumentError("retain must be positive"))
    generations=database_snapshot_generations(path)
    current_generation=current_database_generation(path)
    kept=String[]

    current_generation===nothing||push!(kept,current_generation)

    for generation in generations
        generation in kept&&continue

        if length(kept)<retain
            push!(kept,generation)
        else
            rm(joinpath(database_snapshot_root(path),generation);recursive=true,force=true,)
        end
    end

    fsync_path(database_snapshot_root(path))
    return nothing
end
