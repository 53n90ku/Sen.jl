mutable struct IDStore
    ids::Vector{Any}
    positions::Dict{Any,Int}
end

function create_id_store(; initial_capacity::Int = 0)
    initial_capacity>=0||throw(ArgumentError("initial capacity cannot be negative"))

    ids=Any[]
    positions=Dict{Any,Int}()

    sizehint!(ids, initial_capacity)
    sizehint!(positions, initial_capacity)

    return IDStore(ids, positions)
end

function insert_id!(store::IDStore, id)
    id===nothing&&throw(ArgumentError("id cannot be nothing"))
    haskey(store.positions, id)&&throw(ArgumentError("id already exists"))

    position=length(store.ids)+1

    push!(store.ids, id)
    store.positions[id]=position

    return position
end

function swap_delete_id!(store::IDStore, id)
    position=get_position(store, id)
    last_position=length(store)
    moved_id=position==last_position ? nothing : store.ids[last_position]
    delete!(store.positions, id)

    if moved_id!==nothing
        store.ids[position]=moved_id
        store.positions[moved_id]=position
    end

    pop!(store.ids)
    return (position = position, moved_id = moved_id)
end

function get_id(store::IDStore, position::Int)
    1<=position<=length(store)||throw(BoundsError(store, position))
    return store.ids[position]
end

function get_position(store::IDStore, id)
    haskey(store.positions, id)||throw(KeyError(id))
    return store.positions[id]
end

function has_id(store::IDStore, id)
    return haskey(store.positions, id)
end

function stored_ids(store::IDStore)
    return store.ids
end

Base.length(store::IDStore) = length(store.ids)

const ID_STORE_MAGIC_V1=UInt8[0x53, 0x45, 0x4e, 0x49, 0x44, 0x53, 0x30, 0x31]
const ID_STORE_MAGIC_V2=UInt8[0x53, 0x45, 0x4e, 0x49, 0x44, 0x53, 0x30, 0x32]

function next_available_id(store::IDStore)
    id=length(store)+1

    while haskey(store.positions, id)
        id+=1
    end

    return id
end

function id_store_path(path::AbstractString, filename::AbstractString = "ids.bin")
    isempty(filename)&&throw(ArgumentError("id filename cannot be empty"))
    basename(filename)==filename||throw(ArgumentError("id filename must be a basename"))
    return joinpath(path, filename)
end

function save_id_store(
    path::AbstractString,
    store::IDStore;
    filename::AbstractString = "ids.bin",
)
    mkpath(path)

    id_path=id_store_path(path, filename)

    open(id_path, "w") do io
        write(io, ID_STORE_MAGIC_V2)
        write_portable_uint16(io, PORTABLE_VALUE_FORMAT_VERSION)
        write_portable_length(io, length(store), "id count")

        for id in store.ids
            write_portable_value(io, id)
        end
    end

    return id_path
end

function load_id_store(path::AbstractString; filename::AbstractString = "ids.bin")
    id_path=id_store_path(path, filename)
    isfile(id_path)||throw(ArgumentError("id file does not exist"))

    return open(id_path, "r") do io
        try
            magic=read(io, length(ID_STORE_MAGIC_V2))

            if magic==ID_STORE_MAGIC_V2
                version=Int(read_portable_uint16(io))
                version==PORTABLE_VALUE_FORMAT_VERSION||throw(
                    ArgumentError("unsupported id format version"),
                )
                count=read_portable_length(io, "id count")
                store=create_id_store(initial_capacity = count)

                for _ = 1:count
                    insert_id!(store, read_portable_value(io))
                end

                eof(io)||throw(ArgumentError("id file contains unexpected data"))
                return store
            elseif magic==ID_STORE_MAGIC_V1
                count=Int(read(io, Int64))
                count>=0||throw(ArgumentError("stored id count cannot be negative"))
                store=create_id_store(initial_capacity = count)

                for _ = 1:count
                    insert_id!(store, Serialization.deserialize(io))
                end

                eof(io)||throw(ArgumentError("id file contains unexpected data"))
                return store
            end

            throw(ArgumentError("invalid id file"))
        catch error
            portable_read_error(error, "id file")
        end
    end
end
