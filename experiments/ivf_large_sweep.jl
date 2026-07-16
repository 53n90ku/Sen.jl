pushfirst!(LOAD_PATH, normpath(joinpath(@__DIR__, "..", "benchmark", "SenBench")))

using SenBench

function integer_env(name::String, default::Int)
    value=parse(Int, get(ENV, name, string(default)))
    value>0||error("$(name) must be positive")
    return value
end

function integer_list_env(name::String, default::Vector{Int})
    values=parse.(Int, split(get(ENV, name, join(default, ',')), ','))
    all(>(0), values)||error("$(name) values must be positive")
    return values
end

function main()
    root=normpath(joinpath(@__DIR__, ".."))
    vector_count=integer_env("SEN_SUBSET_COUNT", 300_000)
    rehearsal=vector_count<300_000
    data_path=get(ENV, "SEN_REAL_DATA", joinpath(root, "data", "arxiv-for-fanns-300k"))
    partition_name=rehearsal ? "sen_query_partitions_rehearsal_$(vector_count)" :
                   "sen_query_partitions_300k_v1"
    partition_path=get(ENV, "SEN_QUERY_PARTITIONS", joinpath(data_path, partition_name))
    output_name=rehearsal ? "ivf-quality-rehearsal-$(vector_count)" : "ivf-quality-300k"
    output_path=get(ENV, "SEN_SWEEP_OUTPUT", joinpath(root, "results", output_name))
    manifest=load_dataset_manifest(joinpath(data_path, "sen_dataset.toml"))
    manifest.vector_count==vector_count||error(
        "prepared dataset vector count does not match SEN_SUBSET_COUNT",
    )
    partitions=load_query_partitions(
        partition_path;
        dataset_hash = manifest.preprocessing_hash,
    )
    development=query_partition_indices(partitions, :development)
    validation=query_partition_indices(partitions, :validation)
    println(
        "loading groundtruth development=$(length(development)) validation=$(length(validation)) confirmation=sealed",
    )
    dataset=load_arxiv_subset(
        data_path,
        development,
        validation;
        k = integer_env("SEN_REAL_K", 10),
        verify_hashes = get(ENV, "SEN_VERIFY_HASHES", "0")=="1",
        groundtruth_block_size = integer_env("SEN_GROUNDTRUTH_BLOCK_SIZE", 2_048),
    )
    results=run_ivf_quality_sweep(
        dataset,
        output_path;
        nlists_values = integer_list_env(
            "SEN_SWEEP_NLISTS",
            rehearsal ? [64, 128] : [1_024, 1_536, 2_048],
        ),
        training_count = integer_env(
            "SEN_SWEEP_TRAINING_COUNT",
            rehearsal ? min(20_000, vector_count) : 50_000,
        ),
        iterations = integer_env("SEN_SWEEP_ITERATIONS", rehearsal ? 4 : 8),
        restarts = integer_env("SEN_SWEEP_RESTARTS", rehearsal ? 1 : 2),
        seed = integer_env("SEN_REAL_SEED", 42),
        force = get(ENV, "SEN_SWEEP_FORCE", "0")=="1",
    )
    println("IVF sweep configurations=$(length(results)) output=$(output_path)")
    return results
end

main()
