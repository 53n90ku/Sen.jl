const DATABASE_SEGMENT_MANIFEST="segments.toml"
const DATABASE_SEGMENT_FORMAT_VERSION=1
const SEGMENT_STATE_MAGIC=UInt8[0x53,0x45,0x4e,0x53,0x54,0x41,0x30,0x31]

function database_segment_manifest_path(path::AbstractString)
    return joinpath(path,DATABASE_SEGMENT_MANIFEST)
end

function checked_segment_filename(value,name::AbstractString)
    filename=String(value)
    isempty(filename)&&throw(ArgumentError("$(name) cannot be empty"))
    basename(filename)==filename||throw(ArgumentError("$(name) must be a basename"))
    return filename
end

function checked_segment_revision(value,name::AbstractString)
    value isa Integer||throw(ArgumentError("$(name) must be an integer"))
    value>=0||throw(ArgumentError("$(name) cannot be negative"))
    UInt128(value)<=UInt128(typemax(UInt64))||throw(ArgumentError("$(name) is too large"))
    return UInt64(value)
end

function save_segment_state(path::AbstractString,filename::AbstractString,excluded::BitVector,tombstone_ids::Set{Any})
    state_path=joinpath(path,checked_segment_filename(filename,"segment state filename"))
    ordered_tombstones=sort!(collect(tombstone_ids);by=value->sprint(show,value),)

    open(state_path,"w") do io
        write(io,SEGMENT_STATE_MAGIC)
        write_portable_uint16(io,DATABASE_SEGMENT_FORMAT_VERSION)
        write_portable_length(io,length(excluded),"segment exclusion count")

        for value in excluded
            write(io,value ? UInt8(1) : UInt8(0))
        end

        write_portable_length(io,length(ordered_tombstones),"segment tombstone count")

        for id in ordered_tombstones
            write_portable_value(io,id)
        end
    end

    return state_path
end

function load_segment_state(path::AbstractString,filename::AbstractString,expected_count::Int)
    state_path=joinpath(path,checked_segment_filename(filename,"segment state filename"))
    isfile(state_path)||throw(ArgumentError("segment state file does not exist"))

    return open(state_path,"r") do io
        try
            read(io,length(SEGMENT_STATE_MAGIC))==SEGMENT_STATE_MAGIC||throw(ArgumentError("invalid segment state file"))
            Int(read_portable_uint16(io))==DATABASE_SEGMENT_FORMAT_VERSION||throw(ArgumentError("unsupported segment state format version"))
            count=read_portable_length(io,"segment exclusion count")
            count==expected_count||throw(DimensionMismatch("segment exclusions do not match vector count"))
            excluded=falses(count)

            for index in 1:count
                value=read(io,UInt8)
                value in (UInt8(0),UInt8(1))||throw(ArgumentError("stored segment exclusion is invalid"))
                excluded[index]=value==UInt8(1)
            end

            tombstone_count=read_portable_length(io,"segment tombstone count")
            tombstone_ids=Set{Any}()

            for _ in 1:tombstone_count
                id=read_portable_value(io)
                id in tombstone_ids&&throw(ArgumentError("stored segment tombstone is duplicated"))
                push!(tombstone_ids,id)
            end

            eof(io)||throw(ArgumentError("segment state file contains unexpected data"))
            return(excluded=excluded,tombstone_ids=tombstone_ids,)
        catch error
            portable_read_error(error,"segment state file")
        end
    end
end

function segment_build_descriptor(config::IndexBuildConfig,index_file::AbstractString)
    return Dict(
        "index_file"=>String(index_file),
        "nlists"=>config.nlists,
        "iterations"=>config.iterations,
        "seed"=>config.seed,
        "restarts"=>config.restarts,
        "training_count"=>config.training_count,
    )
end

function load_segment_build_config(entry::AbstractDict,count::Int)
    for key in ("nlists","iterations","seed","restarts","training_count")
        haskey(entry,key)||throw(ArgumentError("segment index descriptor is missing $(key)"))
    end

    return IndexBuildConfig(
        Int(entry["nlists"]),
        count;
        iterations=Int(entry["iterations"]),
        seed=Int(entry["seed"]),
        restarts=Int(entry["restarts"]),
        training_count=Int(entry["training_count"]),
    )
end

function save_database_segments(path::AbstractString,db::VectorDB)
    db.segment_mode||return nothing
    isempty(db.immutable_segments)&&error("segmented database has no immutable primary segment")
    entries=Vector{Dict{String,Any}}(undef,length(db.immutable_segments))

    for(segment_number,segment) in enumerate(db.immutable_segments)
        prefix="segment-$(lpad(segment_number,6,'0'))"
        vector_file="$(prefix)-vectors.bin"
        metadata_file="$(prefix)-metadata.bin"
        id_file="$(prefix)-ids.bin"
        state_file="$(prefix)-state.bin"
        save_vector_store(path,segment.vector_store;filename=vector_file,)
        save_metadata_store(path,segment.metadata_store;filename=metadata_file,)
        save_id_store(path,segment.id_store;filename=id_file,)
        save_segment_state(path,state_file,segment.excluded,segment.tombstone_ids)
        entry=Dict{String,Any}(
            "id"=>segment.id,
            "revision_start"=>Int(segment.revision_start),
            "revision_end"=>Int(segment.revision_end),
            "vector_file"=>vector_file,
            "metadata_file"=>metadata_file,
            "id_file"=>id_file,
            "state_file"=>state_file,
        )

        if segment.index!==nothing
            config=segment.build_config
            config===nothing&&error("indexed segment has no build configuration")
            index_file="$(prefix)-index.bin"
            save_ivf_index(path,segment.index.ivf;filename=index_file,)
            merge!(entry,segment_build_descriptor(config,index_file))
        end

        entries[segment_number]=entry
    end

    active=db.active_segment
    active_prefix="active-segment"
    active_vector_file="$(active_prefix)-vectors.bin"
    active_metadata_file="$(active_prefix)-metadata.bin"
    active_id_file="$(active_prefix)-ids.bin"
    active_state_file="$(active_prefix)-state.bin"
    save_vector_store(path,active.store.vector_store;filename=active_vector_file,)
    save_metadata_store(path,active.store.metadata_store;filename=active_metadata_file,)
    save_id_store(path,active.store.id_store;filename=active_id_file,)
    save_segment_state(path,active_state_file,falses(0),active.tombstone_ids)
    data=Dict{String,Any}(
        "format_version"=>DATABASE_SEGMENT_FORMAT_VERSION,
        "database_revision"=>Int(db.revision),
        "immutable_segments"=>entries,
        "active_segment"=>Dict(
            "id"=>active.id,
            "revision_start"=>Int(active.revision_start),
            "revision_end"=>Int(active.revision_end),
            "vector_file"=>active_vector_file,
            "metadata_file"=>active_metadata_file,
            "id_file"=>active_id_file,
            "state_file"=>active_state_file,
        ),
    )
    db.index_revision===nothing||(data["index_revision"]=Int(db.index_revision))
    manifest_path=database_segment_manifest_path(path)

    open(manifest_path,"w") do io
        TOML.print(io,data)
    end

    return manifest_path
end

function load_immutable_segment(path::AbstractString,entry::AbstractDict,dim::Int,metric::Symbol;mmap_vectors::Union{Bool,Symbol},mmap_threshold_bytes::Int,)
    required=("id","revision_start","revision_end","vector_file","metadata_file","id_file","state_file")
    all(key->haskey(entry,key),required)||throw(ArgumentError("immutable segment descriptor is incomplete"))
    vector_file=checked_segment_filename(entry["vector_file"],"segment vector filename")
    metadata_file=checked_segment_filename(entry["metadata_file"],"segment metadata filename")
    id_file=checked_segment_filename(entry["id_file"],"segment id filename")
    vector_store=load_vector_store(path;filename=vector_file,mmap=mmap_vectors,mmap_threshold_bytes=mmap_threshold_bytes,)
    metadata_store=load_metadata_store(path;filename=metadata_file,)
    id_store=load_id_store(path;filename=id_file,)
    vector_store.dim==dim||throw(DimensionMismatch("segment vector dimension does not match database"))
    count=length(vector_store)
    count==length(metadata_store)==length(id_store)||throw(DimensionMismatch("stored segment stores are misaligned"))
    validate_stored_vectors!(stored_vectors(vector_store),metric)
    state=load_segment_state(path,String(entry["state_file"]),count)
    index=nothing
    filter_index=build_bitset_index(stored_metadata(metadata_store))
    build_config=nothing
    index_bytes=Base.summarysize(filter_index)

    if haskey(entry,"index_file")
        index_file=checked_segment_filename(entry["index_file"],"segment index filename")
        ivf=load_ivf_index(path;filename=index_file,)
        size(ivf.centroids,1)==dim||throw(DimensionMismatch("segment index dimension does not match database"))
        sum(length,ivf.lists)==count||throw(DimensionMismatch("segment index count does not match vector count"))
        ivf.metric===metric||throw(ArgumentError("segment index metric does not match database"))
        build_config=load_segment_build_config(entry,count)
        length(ivf.lists)==build_config.nlists||throw(DimensionMismatch("segment index list count does not match build configuration"))
        index=build_filter_aware_ivf(ivf,stored_metadata(metadata_store))
        index_bytes=Base.summarysize((index,filter_index,))
    end

    return create_immutable_segment(
        String(entry["id"]),
        checked_segment_revision(entry["revision_start"],"segment revision start"),
        checked_segment_revision(entry["revision_end"],"segment revision end"),
        vector_store,
        metadata_store,
        id_store;
        excluded=state.excluded,
        tombstone_ids=state.tombstone_ids,
        index=index,
        filter_index=filter_index,
        build_config=build_config,
        index_bytes=index_bytes,
    )
end

function load_active_segment(path::AbstractString,entry::AbstractDict,dim::Int,metric::Symbol)
    required=("id","revision_start","revision_end","vector_file","metadata_file","id_file","state_file")
    all(key->haskey(entry,key),required)||throw(ArgumentError("active segment descriptor is incomplete"))
    vector_store=load_vector_store(path;filename=checked_segment_filename(entry["vector_file"],"active vector filename"),mmap=false,)
    metadata_store=load_metadata_store(path;filename=checked_segment_filename(entry["metadata_file"],"active metadata filename"),)
    id_store=load_id_store(path;filename=checked_segment_filename(entry["id_file"],"active id filename"),)
    vector_store.dim==dim||throw(DimensionMismatch("active segment vector dimension does not match database"))
    count=length(vector_store)
    count==length(metadata_store)==length(id_store)||throw(DimensionMismatch("stored active segment stores are misaligned"))
    validate_stored_vectors!(stored_vectors(vector_store),metric)
    state=load_segment_state(path,String(entry["state_file"]),0)
    store=DeltaStore(vector_store,metadata_store,id_store)
    return ActiveSegment(
        String(entry["id"]),
        checked_segment_revision(entry["revision_start"],"active segment revision start"),
        checked_segment_revision(entry["revision_end"],"active segment revision end"),
        store,
        state.tombstone_ids,
    )
end

function validate_loaded_segment_order(segments::Vector{ImmutableSegment},active::ActiveSegment,revision::UInt64)
    isempty(segments)&&throw(ArgumentError("segment manifest has no immutable primary segment"))
    first(segments).revision_start==UInt64(0)||throw(ArgumentError("primary segment must begin at revision zero"))
    previous_end=first(segments).revision_end

    for segment in Iterators.drop(segments,1)
        segment.revision_start==next_segment_revision(previous_end)||throw(ArgumentError("immutable segment revisions are not contiguous"))
        previous_end=segment.revision_end
    end

    if active_segment_is_empty(active)
        active.revision_start==next_segment_revision(previous_end)||throw(ArgumentError("empty active segment does not follow immutable segments"))
        active.revision_end==revision||throw(ArgumentError("empty active segment revision does not match database"))
        previous_end==revision||throw(ArgumentError("immutable segments do not reach database revision"))
    else
        active.revision_start==next_segment_revision(previous_end)||throw(ArgumentError("active segment does not follow immutable segments"))
        active.revision_end==revision||throw(ArgumentError("active segment does not reach database revision"))
    end

    return true
end

function load_database_segments(path::AbstractString,dim::Int,metric::Symbol,expected_revision::UInt64;mmap_vectors::Union{Bool,Symbol}=:auto,mmap_threshold_bytes::Int=DEFAULT_VECTOR_MMAP_THRESHOLD_BYTES,)
    manifest_path=database_segment_manifest_path(path)
    isfile(manifest_path)||return nothing
    data=TOML.parsefile(manifest_path)

    for key in ("format_version","database_revision","immutable_segments","active_segment")
        haskey(data,key)||throw(ArgumentError("segment manifest is missing $(key)"))
    end

    Int(data["format_version"])==DATABASE_SEGMENT_FORMAT_VERSION||throw(ArgumentError("unsupported segment manifest format version"))
    revision=checked_segment_revision(data["database_revision"],"segment database revision")
    revision==expected_revision||throw(ArgumentError("segment manifest revision does not match database manifest"))
    entries=data["immutable_segments"]
    entries isa AbstractVector||throw(ArgumentError("immutable segment table is invalid"))
    segments=ImmutableSegment[
        load_immutable_segment(path,entry,dim,metric;mmap_vectors=mmap_vectors,mmap_threshold_bytes=mmap_threshold_bytes,)
        for entry in entries
    ]
    active_entry=data["active_segment"]
    active_entry isa AbstractDict||throw(ArgumentError("active segment table is invalid"))
    active=load_active_segment(path,active_entry,dim,metric)
    validate_active_segment(active,dim,revision)
    validate_loaded_segment_order(segments,active,revision)
    index_revision=haskey(data,"index_revision") ? checked_segment_revision(data["index_revision"],"segment index revision") : nothing
    primary=first(segments)
    (primary.index===nothing)===(index_revision===nothing)||throw(ArgumentError("segment index revision does not match primary index"))
    index_revision===nothing||index_revision<=revision||throw(ArgumentError("segment index revision is ahead of database"))
    length(Set(segment.id for segment in segments))==length(segments)||throw(ArgumentError("immutable segment ids must be unique"))
    active.id in Set(segment.id for segment in segments)&&throw(ArgumentError("active segment id duplicates an immutable segment id"))
    return(immutable_segments=segments,active_segment=active,index_revision=index_revision,)
end
