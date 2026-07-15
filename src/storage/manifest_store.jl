using TOML

struct IndexBuildConfig
    nlists::Int
    iterations::Int
    seed::Int
    restarts::Int
    training_count::Int
end

function IndexBuildConfig(nlists::Int,count::Int;iterations::Int=20,seed::Int=42,restarts::Int=1,training_count::Int=count,)
    count>0||throw(ArgumentError("database cannot be empty"))
    1<=nlists<=count||throw(ArgumentError("nlists must be between 1 and vector count"))
    iterations>0||throw(ArgumentError("iterations must be positive"))
    restarts>0||throw(ArgumentError("restarts must be positive"))
    nlists<=training_count<=count||throw(ArgumentError("training count must be between nlists and vector count"))
    return IndexBuildConfig(nlists,iterations,seed,restarts,training_count)
end

struct DatabaseManifest
    format_version::Int
    dim::Int
    metric::Symbol
    count::Int
    revision::UInt64
    index_revision::Union{Nothing,UInt64}
    build_config::Union{Nothing,IndexBuildConfig}
end

function Base.getproperty(manifest::DatabaseManifest,name::Symbol)
    if name===:nlists
        config=getfield(manifest,:build_config)
        return config===nothing ? nothing : config.nlists
    end

    return getfield(manifest,name)
end

function create_database_manifest(dim::Int,metric::Symbol,count::Int;format_version::Int=1,nlists::Union{Nothing,Int}=nothing,revision::Integer=count,index_revision::Union{Nothing,Integer}=nlists===nothing ? nothing : revision,iterations::Int=20,seed::Int=42,restarts::Int=1,training_count::Int=count,)
    format_version in (1,2)||throw(ArgumentError("unsupported manifest format version"))
    dim>0||throw(ArgumentError("dimension must be positive"))
    metric in (:cosine,:dot)||throw(ArgumentError("metric must be cosine or dot"))
    count>=0||throw(ArgumentError("count cannot be negative"))
    revision>=0||throw(ArgumentError("revision cannot be negative"))
    index_revision===nothing||index_revision>=0||throw(ArgumentError("index revision cannot be negative"))
    index_revision===nothing||index_revision<=revision||throw(ArgumentError("index revision cannot exceed database revision"))
    index_revision===nothing||nlists!==nothing||throw(ArgumentError("index revision requires an index build config"))

    config=nlists===nothing ? nothing : IndexBuildConfig(nlists,count;iterations=iterations,seed=seed,restarts=restarts,training_count=training_count,)
    return DatabaseManifest(format_version,dim,metric,count,UInt64(revision),index_revision===nothing ? nothing : UInt64(index_revision),config)
end

function save_manifest(path::AbstractString,manifest::DatabaseManifest)
    mkpath(path)
    manifest_path=joinpath(path,"manifest.toml")
    data=Dict(
        "format_version"=>manifest.format_version,
        "dimension"=>manifest.dim,
        "metric"=>String(manifest.metric),
        "vector_count"=>manifest.count,
    )

    if manifest.format_version>=2
        data["revision"]=Int(manifest.revision)
        manifest.index_revision===nothing||setindex!(data,Int(manifest.index_revision),"index_revision")
    end

    if manifest.build_config!==nothing
        config=manifest.build_config
        data["nlists"]=config.nlists

        if manifest.format_version>=2
            data["iterations"]=config.iterations
            data["seed"]=config.seed
            data["restarts"]=config.restarts
            data["training_count"]=config.training_count
        end
    end

    open(manifest_path,"w") do io
        TOML.print(io,data)
    end

    return manifest_path
end

function load_manifest(path::AbstractString)
    manifest_path=joinpath(path,"manifest.toml")
    isfile(manifest_path)||throw(ArgumentError("manifest file does not exist"))
    data=TOML.parsefile(manifest_path)

    for key in ("format_version","dimension","metric","vector_count")
        haskey(data,key)||throw(ArgumentError("manifest is missing required fields"))
    end

    format_version=Int(data["format_version"])
    count=Int(data["vector_count"])
    nlists=haskey(data,"nlists") ? Int(data["nlists"]) : nothing
    format_version>=2&&!haskey(data,"revision")&&throw(ArgumentError("manifest is missing revision"))
    revision=format_version>=2 ? Int(data["revision"]) : count
    index_revision=if format_version>=2
        haskey(data,"index_revision") ? Int(data["index_revision"]) : nothing
    else
        nlists===nothing ? nothing : revision
    end

    return create_database_manifest(
        Int(data["dimension"]),
        Symbol(data["metric"]),
        count;
        format_version=format_version,
        nlists=nlists,
        revision=revision,
        index_revision=index_revision,
        iterations=Int(get(data,"iterations",20)),
        seed=Int(get(data,"seed",42)),
        restarts=Int(get(data,"restarts",1)),
        training_count=Int(get(data,"training_count",count)),
    )
end
