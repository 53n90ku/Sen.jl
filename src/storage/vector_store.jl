mutable struct VectorStore
    dim::Int
    vectors::Matrix{Float32}
    count::Int
end

function create_vector_store(dim::Int;initial_capacity::Int=0,)
    dim>0||throw(ArgumentError("dimension must be positive"))
    initial_capacity>=0||throw(ArgumentError("initial capacity cannot be negative"))

    vectors=Matrix{Float32}(undef,dim,initial_capacity)
    return VectorStore(dim,vectors,0)
end

function ensure_capacity!(store::VectorStore,required::Int)
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
    1<=index<=store.count||throw(BoundsError(store,index))
    length(vector)==store.dim||throw(DimensionMismatch("vector dimension doesnt match store"))
    converted=Float32[dimension for dimension in vector]

    for dimension in 1:store.dim
        store.vectors[dimension,index]=converted[dimension]
    end

    return store
end

function swap_delete_vector!(store::VectorStore,index::Int)
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

const VECTOR_STORE_MAGIC=UInt8[0x53,0x45,0x4e,0x56,0x45,0x43,0x30,0x31]

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

function load_vector_store(path::AbstractString)
    vector_path=joinpath(path,"vectors.bin")
    isfile(vector_path)||throw(ArgumentError("vector file does not exist"))

    return open(vector_path,"r") do io
        magic=Vector{UInt8}(undef,length(VECTOR_STORE_MAGIC))
        read!(io,magic)
        magic==VECTOR_STORE_MAGIC||throw(ArgumentError("invalid vector file"))

        dim=Int(read(io,Int64))
        count=Int(read(io,Int64))

        dim>0||throw(ArgumentError("stored dimension must be positive"))
        count>=0||throw(ArgumentError("stored count cannot be negative"))

        vectors=Matrix{Float32}(undef,dim,count)
        read!(io,vectors)
        eof(io)||throw(ArgumentError("vector file contains unexpected data"))

        return VectorStore(dim,vectors,count)
    end
end
