pushfirst!(LOAD_PATH,normpath(joinpath(@__DIR__,"..","benchmark","SenBench")))

using SenBench

function integer_env(name::String,default::Int)
    value=parse(Int,get(ENV,name,string(default)))
    value>0||error("$(name) must be positive")
    return value
end

function partition_counts(vector_count::Int)
    rehearsal=vector_count<300_000
    return(
        development=integer_env("SEN_DEVELOPMENT_QUERIES",rehearsal ? 48 : 256),
        validation=integer_env("SEN_VALIDATION_QUERIES",rehearsal ? 24 : 128),
        confirmation=integer_env("SEN_CONFIRMATION_QUERIES",rehearsal ? 48 : 256),
    )
end

function main()
    root=normpath(joinpath(@__DIR__,".."))
    vector_count=integer_env("SEN_SUBSET_COUNT",300_000)
    data_path=get(ENV,"SEN_REAL_DATA",joinpath(root,"data","arxiv-for-fanns-300k"))
    seed=integer_env("SEN_REAL_SEED",42)
    chunk_records=integer_env("SEN_DOWNLOAD_CHUNK_RECORDS",4_096)
    println("preparing subset vectors=$(vector_count) path=$(data_path)")
    prepared=prepare_arxiv_subset(data_path;vector_count=vector_count,chunk_records=chunk_records,seed=seed,)
    manifest=prepared.manifest
    counts=partition_counts(vector_count)
    partition_name=vector_count<300_000 ? "sen_query_partitions_rehearsal_$(vector_count)" : "sen_query_partitions_300k_v1"
    partition_path=get(ENV,"SEN_QUERY_PARTITIONS",joinpath(data_path,partition_name))

    if !isfile("$(partition_path).toml")
        labels=load_arxiv_query_labels(joinpath(data_path,"em_query_attributes.jsonl"))
        metadata=load_arxiv_metadata(joinpath(data_path,"database_attributes.jsonl"))
        filters=NamedTuple[(label=label,) for label in labels]
        selectivities=dataset_selectivities(metadata,filters)
        strata=Int[selectivity_bucket(value)===:rare ? 1 : selectivity_bucket(value)===:medium ? 2 : 3 for value in selectivities]
        partitions=balanced_query_partitions(
            labels;
            development_count=counts.development,
            validation_count=counts.validation,
            confirmation_count=counts.confirmation,
            seed=seed,
            strata=strata,
        )
        save_query_partitions(partition_path,partitions;dataset_hash=manifest.preprocessing_hash,query_count=manifest.query_count,)
        println("sealed query partitions path=$(partition_path).toml")
    end

    partitions=load_query_partitions(partition_path;dataset_hash=manifest.preprocessing_hash,)
    development=query_partition_indices(partitions,:development)
    validation=query_partition_indices(partitions,:validation)
    println("prepared vectors=$(manifest.vector_count) development=$(length(development)) validation=$(length(validation)) confirmation=sealed")
    println("dataset_hash=$(manifest.preprocessing_hash)")
    return prepared
end

main()
