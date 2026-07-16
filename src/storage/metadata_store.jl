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

const METADATA_STORE_MAGIC_V1=UInt8[0x53,0x45,0x4e,0x4d,0x45,0x54,0x30,0x31]
const METADATA_STORE_MAGIC_V2=UInt8[0x53,0x45,0x4e,0x4d,0x45,0x54,0x30,0x32]

function metadata_store_path(path::AbstractString,filename::AbstractString="metadata.bin")
    isempty(filename)&&throw(ArgumentError("metadata filename cannot be empty"))
    basename(filename)==filename||throw(ArgumentError("metadata filename must be a basename"))
    return joinpath(path,filename)
end

function save_metadata_store(path::AbstractString,store::MetadataStore;filename::AbstractString="metadata.bin",)
    mkpath(path)

    metadata_path=metadata_store_path(path,filename)

    open(metadata_path,"w") do io
        write(io,METADATA_STORE_MAGIC_V2)
        write_portable_uint16(io,PORTABLE_VALUE_FORMAT_VERSION)
        write_portable_length(io,length(store),"metadata count")

        for metadata in store.metadata
            write_portable_named_tuple(io,metadata)
        end
    end

    return metadata_path
end

function load_metadata_store(path::AbstractString;filename::AbstractString="metadata.bin",)
    metadata_path=metadata_store_path(path,filename)
    isfile(metadata_path)||throw(ArgumentError("metadata file does not exist"))

    return open(metadata_path,"r") do io
        try
            magic=read(io,length(METADATA_STORE_MAGIC_V2))

            if magic==METADATA_STORE_MAGIC_V2
                version=Int(read_portable_uint16(io))
                version==PORTABLE_VALUE_FORMAT_VERSION||throw(ArgumentError("unsupported metadata format version"))
                count=read_portable_length(io,"metadata count")
                metadata=Vector{NamedTuple}(undef,count)

                for index in 1:count
                    metadata[index]=read_portable_named_tuple(io)
                end

                eof(io)||throw(ArgumentError("metadata file contains unexpected data"))
                return MetadataStore(metadata)
            elseif magic==METADATA_STORE_MAGIC_V1
                count=Int(read(io,Int64))
                count>=0||throw(ArgumentError("stored metadata count cannot be negative"))
                metadata=Vector{NamedTuple}(undef,count)

                for index in 1:count
                    value=Serialization.deserialize(io)
                    value isa NamedTuple||throw(ArgumentError("stored metadata must be a named tuple"))
                    metadata[index]=value
                end

                eof(io)||throw(ArgumentError("metadata file contains unexpected data"))
                return MetadataStore(metadata)
            end

            throw(ArgumentError("invalid metadata file"))
        catch error
            portable_read_error(error,"metadata file")
        end
    end
end
