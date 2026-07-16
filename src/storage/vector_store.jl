using Mmap

mutable struct VectorStore
    dim::Int
    vectors::Matrix{Float32}
    count::Int
    mapped::Bool
end

VectorStore(dim::Int,vectors::Matrix{Float32},count::Int)=VectorStore(dim,vectors,count,false)

function create_vector_store(dim::Int;initial_capacity::Int=0,)
    dim>0||throw(ArgumentError("dimension must be positive"))
    initial_capacity>=0||throw(ArgumentError("initial capacity cannot be negative"))

    vectors=Matrix{Float32}(undef,dim,initial_capacity)
    return VectorStore(dim,vectors,0)
end

function ensure_capacity!(store::VectorStore,required::Int)
    store.mapped&&throw(ArgumentError("memory mapped vector store is read only"))
    required<=size(store.vectors,2)&&return store

    current_capacity=size(store.vectors,2)
    new_capacity=max(required,max(1,current_capacity*2))
    resized_vectors=Matrix{Float32}(undef,store.dim,new_capacity)

    if store.count>0
        resized_vectors[:,1:store.count].=store.vectors[:,1:store.count]
    end

    store.vectors=resized_vectors
    return store
end

function insert_vector!(store::VectorStore,vector::AbstractVector{<:Real})
    store.mapped&&throw(ArgumentError("memory mapped vector store is read only"))
    length(vector)==store.dim||throw(DimensionMismatch("vector dimension doesnt match store"))

    new_index=store.count+1
    ensure_capacity!(store,new_index)

    for dimension in 1:store.dim
        store.vectors[dimension,new_index]=Float32(vector[dimension])
    end

    store.count=new_index
    return new_index
end

function update_vector!(store::VectorStore,index::Int,vector::AbstractVector{<:Real})
    store.mapped&&throw(ArgumentError("memory mapped vector store is read only"))
    1<=index<=store.count||throw(BoundsError(store,index))
    length(vector)==store.dim||throw(DimensionMismatch("vector dimension doesnt match store"))
    converted=Float32[dimension for dimension in vector]

    for dimension in 1:store.dim
        store.vectors[dimension,index]=converted[dimension]
    end

    return store
end

function swap_delete_vector!(store::VectorStore,index::Int)
    store.mapped&&throw(ArgumentError("memory mapped vector store is read only"))
    1<=index<=store.count||throw(BoundsError(store,index))
    last_index=store.count

    if index!=last_index
        for dimension in 1:store.dim
            store.vectors[dimension,index]=store.vectors[dimension,last_index]
        end
    end

    store.count-=1
    return store
end

function get_vector(store::VectorStore,index::Int)
    1<=index<=store.count||throw(BoundsError(store,index))
    return @view store.vectors[:,index]
end

function stored_vectors(store::VectorStore)
    return @view store.vectors[:,1:store.count]
end

Base.length(store::VectorStore)=store.count

function is_mapped(store::VectorStore)
    return store.mapped
end

function make_vector_store_writable!(store::VectorStore)
    store.mapped||return store
    vectors=Matrix{Float32}(undef,store.dim,store.count)

    if store.count>0
        copyto!(vectors,stored_vectors(store))
    end

    store.vectors=vectors
    store.mapped=false
    return store
end

function release_vector_store_mapping!(store::VectorStore)
    store.mapped||return store
    store.vectors=Matrix{Float32}(undef,store.dim,0)
    store.count=0
    store.mapped=false
    return store
end

const VECTOR_STORE_MAGIC=UInt8[0x53,0x45,0x4e,0x56,0x45,0x43,0x30,0x31]
const VECTOR_STORE_HEADER_BYTES=length(VECTOR_STORE_MAGIC)+sizeof(Int64)+sizeof(Int64)
const DEFAULT_VECTOR_MMAP_THRESHOLD_BYTES=64*1024*1024

function read_vector_store_header(io::IO,vector_path::AbstractString)
    magic=Vector{UInt8}(undef,length(VECTOR_STORE_MAGIC))

    try
        read!(io,magic)
        magic==VECTOR_STORE_MAGIC||throw(ArgumentError("invalid vector file"))

        dim=Int(read(io,Int64))
        count=Int(read(io,Int64))
        dim>0||throw(ArgumentError("stored dimension must be positive"))
        count>=0||throw(ArgumentError("stored count cannot be negative"))

        value_count=Base.checked_mul(dim,count)
        payload_bytes=Base.checked_mul(value_count,sizeof(Float32))
        expected_bytes=Base.checked_add(VECTOR_STORE_HEADER_BYTES,payload_bytes)
        filesize(vector_path)==expected_bytes||throw(ArgumentError("vector file size does not match its header"))
        return(dim=dim,count=count,payload_bytes=payload_bytes,)
    catch error
        error isa EOFError&&throw(ArgumentError("vector file header is truncated"))
        error isa OverflowError&&throw(ArgumentError("vector file dimensions are too large"))
        rethrow()
    end
end

function should_mmap_vector_store(mode::Union{Bool,Symbol},payload_bytes::Int,threshold_bytes::Int)
    threshold_bytes>=0||throw(ArgumentError("mmap threshold bytes cannot be negative"))
    mode===:auto||mode isa Bool||throw(ArgumentError("mmap must be true, false or :auto"))
    payload_bytes==0&&return false
    return mode===true||(mode===:auto&&payload_bytes>=threshold_bytes)
end

function save_vector_store(path::AbstractString,store::VectorStore)
    mkpath(path)

    vector_path=joinpath(path,"vectors.bin")

    open(vector_path,"w") do io
        write(io,VECTOR_STORE_MAGIC)
        write(io,Int64(store.dim))
        write(io,Int64(store.count))
        write(io,stored_vectors(store))
    end

    return vector_path
end

function load_vector_store(path::AbstractString;mmap::Union{Bool,Symbol}=:auto,mmap_threshold_bytes::Int=DEFAULT_VECTOR_MMAP_THRESHOLD_BYTES,)
    vector_path=joinpath(path,"vectors.bin")
    isfile(vector_path)||throw(ArgumentError("vector file does not exist"))

    return open(vector_path,"r") do io
        header=read_vector_store_header(io,vector_path)

        if should_mmap_vector_store(mmap,header.payload_bytes,mmap_threshold_bytes)
            try
                vectors=Mmap.mmap(io,Matrix{Float32},(header.dim,header.count),VECTOR_STORE_HEADER_BYTES;grow=false,shared=true,)
                return VectorStore(header.dim,vectors,header.count,true)
            catch
                mmap===true&&rethrow()
            end
        end

        seek(io,VECTOR_STORE_HEADER_BYTES)
        vectors=Matrix{Float32}(undef,header.dim,header.count)
        read!(io,vectors)
        return VectorStore(header.dim,vectors,header.count,false)
    end
end
