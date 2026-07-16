const IVF_INDEX_MAGIC=UInt8[0x53,0x45,0x4e,0x49,0x44,0x58,0x30,0x31]
const IVF_INDEX_VERSION=2

function index_file_path(path::AbstractString,filename::AbstractString="index.bin")
    isempty(filename)&&throw(ArgumentError("index filename cannot be empty"))
    basename(filename)==filename||throw(ArgumentError("index filename must be a basename"))
    return joinpath(path,filename)
end

function save_ivf_index(path::AbstractString,index::IVFIndex;filename::AbstractString="index.bin",)
    mkpath(path)
    index_path=index_file_path(path,filename)
    dim,list_count=size(index.centroids)
    vector_count=sum(length,index.lists)
    index.metric in (:cosine,:dot)||throw(ArgumentError("index metric must be cosine or dot"))
    index.routing in (:cosine,:dot,:euclidean)||throw(ArgumentError("index routing must be cosine, dot or euclidean"))
    index.metric===:cosine&&index.routing!==:cosine&&throw(ArgumentError("cosine index must use cosine routing"))
    index.metric===:dot&&!(index.routing in (:dot,:euclidean))&&throw(ArgumentError("dot index routing is invalid"))
    length(index.vector_norms)==vector_count||throw(DimensionMismatch("vector norm count doesnt match index"))
    length(index.list_radii)==list_count||throw(DimensionMismatch("radius count doesnt match index"))

    open(index_path,"w") do io
        write(io,IVF_INDEX_MAGIC)
        write(io,Int64(IVF_INDEX_VERSION))
        write(io,index.metric===:cosine ? UInt8(1) : UInt8(2))
        write(io,index.routing===:cosine ? UInt8(1) : index.routing===:dot ? UInt8(2) : UInt8(3))
        write(io,Int64(dim))
        write(io,Int64(list_count))
        write(io,Int64(vector_count))
        write(io,index.centroids)
        write(io,index.vector_norms)
        write(io,index.list_radii)

        for list in index.lists
            write(io,Int64(length(list)))

            for vector_index in list
                write(io,Int64(vector_index))
            end
        end
    end

    return index_path
end

function load_ivf_index(path::AbstractString;filename::AbstractString="index.bin",)
    index_path=index_file_path(path,filename)
    isfile(index_path)||throw(ArgumentError("index file does not exist"))

    return open(index_path,"r") do io
        magic=Vector{UInt8}(undef,length(IVF_INDEX_MAGIC))
        read!(io,magic)
        magic==IVF_INDEX_MAGIC||throw(ArgumentError("invalid index file"))

        version=Int(read(io,Int64))
        version in (1,IVF_INDEX_VERSION)||throw(ArgumentError("unsupported index format version"))
        metric_code=read(io,UInt8)
        metric=metric_code==1 ? :cosine : metric_code==2 ? :dot : throw(ArgumentError("invalid stored index metric"))
        routing=if version>=2
            routing_code=read(io,UInt8)
            routing_code==1 ? :cosine : routing_code==2 ? :dot : routing_code==3 ? :euclidean : throw(ArgumentError("invalid stored index routing"))
        else
            metric===:cosine ? :cosine : :euclidean
        end
        metric===:cosine&&routing!==:cosine&&throw(ArgumentError("stored cosine index routing is invalid"))
        metric===:dot&&!(routing in (:dot,:euclidean))&&throw(ArgumentError("stored dot index routing is invalid"))
        dim=Int(read(io,Int64))
        list_count=Int(read(io,Int64))
        vector_count=Int(read(io,Int64))

        dim>0||throw(ArgumentError("stored index dimension must be positive"))
        list_count>0||throw(ArgumentError("stored list count must be positive"))
        vector_count>=list_count||throw(ArgumentError("stored vector count cannot be smaller than list count"))

        centroids=Matrix{Float32}(undef,dim,list_count)
        vector_norms=Vector{Float32}(undef,vector_count)
        list_radii=Vector{Float32}(undef,list_count)
        read!(io,centroids)
        read!(io,vector_norms)
        read!(io,list_radii)

        all(isfinite,centroids)||throw(ArgumentError("stored centroids must be finite"))
        all(norm->isfinite(norm)&&norm>0,vector_norms)||throw(ArgumentError("stored vector norms must be positive and finite"))
        all(radius->isfinite(radius)&&0<=radius<=Float32(pi),list_radii)||throw(ArgumentError("stored list radii are invalid"))

        lists=Vector{Vector{Int}}(undef,list_count)
        seen=falses(vector_count)

        for list_index in 1:list_count
            list_size=Int(read(io,Int64))
            list_size>=0||throw(ArgumentError("stored list size cannot be negative"))
            list=Vector{Int}(undef,list_size)

            for position in 1:list_size
                vector_index=Int(read(io,Int64))
                1<=vector_index<=vector_count||throw(ArgumentError("stored vector index is out of bounds"))
                seen[vector_index]&&throw(ArgumentError("stored vector index is duplicated"))
                seen[vector_index]=true
                list[position]=vector_index
            end

            lists[list_index]=list
        end

        all(seen)||throw(ArgumentError("stored index does not contain every vector"))
        eof(io)||throw(ArgumentError("index file contains unexpected data"))

        return IVFIndex(centroids,lists,vector_norms,metric,routing,list_radii,cos.(list_radii),sin.(list_radii))
    end
end

function remove_ivf_index(path::AbstractString;filename::AbstractString="index.bin",)
    index_path=index_file_path(path,filename)
    isfile(index_path)&&rm(index_path)
    return nothing
end
