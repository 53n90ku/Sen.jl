pushfirst!(LOAD_PATH,normpath(joinpath(@__DIR__,"..","benchmark","SenBench")))

using SenBench

function integer_env(name::String,default::Int)
    value=parse(Int,get(ENV,name,string(default)))
    value>0||error("$(name) must be positive")
    return value
end

function integer_list_env(name::String,default::Vector{Int})
    value=get(ENV,name,join(default,','))
    values=parse.(Int,split(value,','))
    all(>(0),values)||error("$(name) values must be positive")
    return values
end

function main()
    root=normpath(joinpath(@__DIR__,".."))
    scale=Symbol(get(ENV,"SEN_REAL_SCALE","medium"))
    data_path=get(ENV,"SEN_REAL_DATA",joinpath(root,"data","arxiv-for-fanns-$(scale)"))
    output_path=get(ENV,"SEN_SWEEP_OUTPUT",joinpath(root,"results","ivf-quality-$(scale)"))
    partition_path=get(ENV,"SEN_QUERY_PARTITIONS",joinpath(data_path,"sen_query_partitions_v3"))
    seed=integer_env("SEN_REAL_SEED",42)
    k=integer_env("SEN_REAL_K",10)
    manifest=SenBench.arxiv_manifest(data_path,scale,seed)

    if !isfile("$(partition_path).toml")
        labels=load_arxiv_query_labels(joinpath(data_path,"em_query_attributes.jsonl"))
        metadata=load_arxiv_metadata(joinpath(data_path,"database_attributes.jsonl"))
        selectivities=dataset_selectivities(metadata,NamedTuple[(label=label,) for label in labels])
        strata=Int[selectivity_bucket(selectivity)===:rare ? 1 : selectivity_bucket(selectivity)===:medium ? 2 : 3 for selectivity in selectivities]
        partitions=balanced_query_partitions(
            labels;
            development_count=integer_env("SEN_DEVELOPMENT_QUERIES",256),
            validation_count=integer_env("SEN_VALIDATION_QUERIES",128),
            confirmation_count=integer_env("SEN_CONFIRMATION_QUERIES",256),
            seed=seed,
            strata=strata,
        )
        save_query_partitions(partition_path,partitions;dataset_hash=manifest.preprocessing_hash,query_count=manifest.query_count,)
        println("sealed query partitions path=$(partition_path).toml")
    end

    partitions=load_query_partitions(partition_path;dataset_hash=manifest.preprocessing_hash,)
    development=query_partition_indices(partitions,:development)
    validation=query_partition_indices(partitions,:validation)
    println("loading development=$(length(development)) validation=$(length(validation)) confirmation=sealed")
    dataset=load_arxiv_fanns_indices(data_path,development,validation;scale=scale,k=k,seed=seed,verify_hashes=get(ENV,"SEN_VERIFY_HASHES","0")=="1",)
    results=run_ivf_quality_sweep(
        dataset,
        output_path;
        nlists_values=integer_list_env("SEN_SWEEP_NLISTS",[64,128,256,512]),
        training_count=integer_env("SEN_SWEEP_TRAINING_COUNT",20_000),
        iterations=integer_env("SEN_SWEEP_ITERATIONS",8),
        restarts=integer_env("SEN_SWEEP_RESTARTS",2),
        seed=seed,
        force=get(ENV,"SEN_SWEEP_FORCE","0")=="1",
    )
    println("IVF sweep configurations=$(length(results)) output=$(output_path)")
    return results
end

main()
