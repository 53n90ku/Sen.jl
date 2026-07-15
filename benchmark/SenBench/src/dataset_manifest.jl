struct DatasetManifest
    name::String
    source::String
    revision::String
    license::String
    vector_count::Int
    dimension::Int
    query_count::Int
    metric::Symbol
    sampling_seed::Int
    files::Dict{String,String}
    file_sizes::Dict{String,Int}
    preprocessing_hash::String
end

function file_sha256(path::AbstractString)
    isfile(path)||throw(ArgumentError("file does not exist: $(path)"))
    return open(path,"r") do io
        bytes2hex(SHA.sha256(io))
    end
end

function manifest_preprocessing_hash(name::AbstractString,revision::AbstractString,vector_count::Int,dimension::Int,query_count::Int,metric::Symbol,sampling_seed::Int,files::AbstractDict)
    entries=join(["$(key)=$(files[key])" for key in sort!(collect(keys(files)))],';')
    value="$(name)|$(revision)|$(vector_count)|$(dimension)|$(query_count)|$(metric)|$(sampling_seed)|$(entries)"
    return bytes2hex(SHA.sha256(value))
end

function create_dataset_manifest(path::AbstractString;name::AbstractString,source::AbstractString,revision::AbstractString="main",license::AbstractString,vector_count::Int,dimension::Int,query_count::Int,metric::Symbol=:cosine,sampling_seed::Int=42,filenames::AbstractVector{<:AbstractString},)
    vector_count>0||throw(ArgumentError("vector count must be positive"))
    dimension>0||throw(ArgumentError("dimension must be positive"))
    query_count>0||throw(ArgumentError("query count must be positive"))
    metric in (:cosine,:dot)||throw(ArgumentError("metric must be cosine or dot"))
    isempty(filenames)&&throw(ArgumentError("manifest files cannot be empty"))

    hashes=Dict{String,String}()
    sizes=Dict{String,Int}()

    for filename in sort!(unique(String.(filenames)))
        file_path=joinpath(path,filename)
        hashes[filename]=file_sha256(file_path)
        sizes[filename]=filesize(file_path)
    end

    preprocessing_hash=manifest_preprocessing_hash(name,revision,vector_count,dimension,query_count,metric,sampling_seed,hashes)
    return DatasetManifest(String(name),String(source),String(revision),String(license),vector_count,dimension,query_count,metric,sampling_seed,hashes,sizes,preprocessing_hash)
end

function dataset_manifest_dict(manifest::DatasetManifest)
    return Dict(
        "name"=>manifest.name,
        "source"=>manifest.source,
        "revision"=>manifest.revision,
        "license"=>manifest.license,
        "vector_count"=>manifest.vector_count,
        "dimension"=>manifest.dimension,
        "query_count"=>manifest.query_count,
        "metric"=>String(manifest.metric),
        "sampling_seed"=>manifest.sampling_seed,
        "files"=>manifest.files,
        "file_sizes"=>manifest.file_sizes,
        "preprocessing_hash"=>manifest.preprocessing_hash,
    )
end

function save_dataset_manifest(path::AbstractString,manifest::DatasetManifest)
    mkpath(dirname(path))
    open(path,"w") do io
        TOML.print(io,dataset_manifest_dict(manifest))
    end
    return String(path)
end

function load_dataset_manifest(path::AbstractString)
    isfile(path)||throw(ArgumentError("dataset manifest does not exist"))
    data=TOML.parsefile(path)
    files=Dict{String,String}(String(key)=>String(value) for(key,value) in data["files"])
    sizes=Dict{String,Int}(String(key)=>Int(value) for(key,value) in data["file_sizes"])
    return DatasetManifest(
        String(data["name"]),
        String(data["source"]),
        String(data["revision"]),
        String(data["license"]),
        Int(data["vector_count"]),
        Int(data["dimension"]),
        Int(data["query_count"]),
        Symbol(data["metric"]),
        Int(data["sampling_seed"]),
        files,
        sizes,
        String(data["preprocessing_hash"]),
    )
end

function validate_dataset_manifest(manifest::DatasetManifest,path::AbstractString;verify_hashes::Bool=true,)
    for(filename,expected_size) in manifest.file_sizes
        file_path=joinpath(path,filename)
        isfile(file_path)||throw(ArgumentError("dataset file is missing: $(filename)"))
        filesize(file_path)==expected_size||throw(ArgumentError("dataset file size changed: $(filename)"))
        verify_hashes||continue
        file_sha256(file_path)==manifest.files[filename]||throw(ArgumentError("dataset file hash changed: $(filename)"))
    end

    expected=manifest_preprocessing_hash(manifest.name,manifest.revision,manifest.vector_count,manifest.dimension,manifest.query_count,manifest.metric,manifest.sampling_seed,manifest.files)
    expected==manifest.preprocessing_hash||throw(ArgumentError("dataset preprocessing hash is invalid"))
    return manifest
end
