using Serialization

mutable struct MetadataStore
    metadata::Vector{NamedTuple}
end

function create_metadata_store(;initial_capacity::Int=0,)
    initial_capacity>=0||throw(ArgumentError("initial capacity cannot be negative"))

    metadata=NamedTuple[]
    sizehint!(metadata,initial_capacity)

    return MetadataStore(metadata)
end

function insert_metadata!(store::MetadataStore,metadata::NamedTuple)
    push!(store.metadata,metadata)
    return length(store.metadata)
end

function update_metadata!(store::MetadataStore,index::Int,metadata::NamedTuple)
    1<=index<=length(store)||throw(BoundsError(store,index))
    store.metadata[index]=metadata
    return store
end

function swap_delete_metadata!(store::MetadataStore,index::Int)
    1<=index<=length(store)||throw(BoundsError(store,index))
    last_index=length(store)
    index==last_index||setindex!(store.metadata,store.metadata[last_index],index)
    pop!(store.metadata)
    return store
end

function get_metadata(store::MetadataStore,index::Int)
    1<=index<=length(store)||throw(BoundsError(store,index))
    return store.metadata[index]
end

function stored_metadata(store::MetadataStore)
    return store.metadata
end

Base.length(store::MetadataStore)=length(store.metadata)

const METADATA_STORE_MAGIC=UInt8[0x53,0x45,0x4e,0x4d,0x45,0x54,0x30,0x31]

function save_metadata_store(path::AbstractString,store::MetadataStore)
    mkpath(path)

    metadata_path=joinpath(path,"metadata.bin")

    open(metadata_path,"w") do io
        write(io,METADATA_STORE_MAGIC)
        write(io,Int64(length(store)))

        for metadata in store.metadata
            serialize(io,metadata)
        end
    end

    return metadata_path
end

function load_metadata_store(path::AbstractString)
    metadata_path=joinpath(path,"metadata.bin")
    isfile(metadata_path)||throw(ArgumentError("metadata file does not exist"))

    return open(metadata_path,"r") do io
        magic=Vector{UInt8}(undef,length(METADATA_STORE_MAGIC))
        read!(io,magic)
        magic==METADATA_STORE_MAGIC||throw(ArgumentError("invalid metadata file"))

        count=Int(read(io,Int64))
        count>=0||throw(ArgumentError("stored metadata count cannot be negative"))

        metadata=Vector{NamedTuple}(undef,count)

        for index in 1:count
            value=deserialize(io)
            value isa NamedTuple||throw(ArgumentError("stored metadata must be a named tuple"))
            metadata[index]=value
        end

        eof(io)||throw(ArgumentError("metadata file contains unexpected data"))

        return MetadataStore(metadata)
    end
end
