pushfirst!(LOAD_PATH, normpath(joinpath(@__DIR__, "..", "benchmark", "SenBench")))

using SenBench
using TOML

function integer_env(name::String, default::Int)
    value=parse(Int, get(ENV, name, string(default)))
    value>0||error("$(name) must be positive")
    return value
end

function freeze_finalist(
    path::String,
    checkpoint::Dict,
    index_file::String,
    partitions::RealQueryPartitions,
    dataset_hash::String;
    nprobe::Int,
    rare_threshold::Float64,
)
    data=Dict(
        "dataset_hash"=>dataset_hash,
        "partition_seed"=>partitions.seed,
        "confirmation_hash"=>partitions.confirmation_hash,
        "configuration_hash"=>String(checkpoint["configuration_hash"]),
        "index_sha256"=>file_sha256(index_file),
        "nlists"=>Int(checkpoint["nlists"]),
        "training_count"=>Int(checkpoint["training_count"]),
        "iterations"=>Int(checkpoint["iterations"]),
        "restarts"=>Int(checkpoint["restarts"]),
        "seed"=>Int(checkpoint["seed"]),
        "nprobe"=>nprobe,
        "rare_threshold"=>rare_threshold,
        "target_recall"=>0.95,
    )

    if isfile(path)
        TOML.parsefile(path)==data||error("frozen finalist configuration changed")
        return data
    end

    temporary="$(path).tmp"
    open(temporary, "w") do io
        TOML.print(io, data)
    end
    mv(temporary, path; force = true)
    return data
end

function print_validation(result)
    println("method\trecall\tp50_ms\tp95_ms\tcandidates_scored")
    for (method, evaluation) in (
        (:exact, result.exact),
        (:ivf_prefilter, result.approximate),
        (:hybrid, result.hybrid),
    )
        summary=evaluation.summary
        println(
            "$(method)\t$(round(summary.average_recall;digits=4,))\t$(round(summary.latency.p50_ms;digits=4,))\t$(round(summary.latency.p95_ms;digits=4,))\t$(round(summary.average_candidates_scored;digits=2,))",
        )
    end
    for (bucket, summary) in
        sort!(collect(result.buckets); by = value->string(first(value)))
        println(
            "hybrid_$(bucket)\t$(round(summary.average_recall;digits=4,))\t$(round(summary.latency.p50_ms;digits=4,))\t$(round(summary.latency.p95_ms;digits=4,))\t$(round(summary.average_candidates_scored;digits=2,))",
        )
    end
end

function main()
    root=normpath(joinpath(@__DIR__, ".."))
    scale=Symbol(get(ENV, "SEN_REAL_SCALE", "medium"))
    data_path=get(ENV, "SEN_REAL_DATA", joinpath(root, "data", "arxiv-for-fanns-$(scale)"))
    sweep_path=get(
        ENV,
        "SEN_SWEEP_OUTPUT",
        joinpath(root, "results", "ivf-quality-$(scale)"),
    )
    champion_path=get(
        ENV,
        "SEN_FINALIST_INDEX",
        joinpath(sweep_path, "lists-512-train-50000-iter-16-restart-3-seed-42"),
    )
    partition_path=get(
        ENV,
        "SEN_QUERY_PARTITIONS",
        joinpath(data_path, "sen_query_partitions_v3"),
    )
    seed=integer_env("SEN_REAL_SEED", 42)
    k=integer_env("SEN_REAL_K", 10)
    nprobe=integer_env("SEN_FINALIST_NPROBE", 128)
    rare_threshold=parse(Float64, get(ENV, "SEN_RARE_THRESHOLD", "0.01"))
    manifest=SenBench.arxiv_manifest(data_path, scale, seed)

    if !isfile("$(partition_path).toml")
        labels=load_arxiv_query_labels(joinpath(data_path, "em_query_attributes.jsonl"))
        metadata=load_arxiv_metadata(joinpath(data_path, "database_attributes.jsonl"))
        selectivities=dataset_selectivities(
            metadata,
            NamedTuple[(label = label,) for label in labels],
        )
        strata=Int[
            selectivity_bucket(selectivity)===:rare ? 1 :
            selectivity_bucket(selectivity)===:medium ? 2 : 3 for
            selectivity in selectivities
        ]
        created=balanced_query_partitions(
            labels;
            development_count = 256,
            validation_count = 128,
            confirmation_count = 256,
            seed = seed,
            strata = strata,
        )
        save_query_partitions(
            partition_path,
            created;
            dataset_hash = manifest.preprocessing_hash,
            query_count = manifest.query_count,
        )
        println("sealed selectivity-balanced query partitions path=$(partition_path).toml")
    end

    partitions=load_query_partitions(
        partition_path;
        dataset_hash = manifest.preprocessing_hash,
    )
    development=query_partition_indices(partitions, :development)
    validation=query_partition_indices(partitions, :validation)
    dataset=load_arxiv_fanns_indices(
        data_path,
        development,
        validation;
        scale = scale,
        k = k,
        seed = seed,
    )
    checkpoint=TOML.parsefile(joinpath(champion_path, "checkpoint.toml"))
    index_file=joinpath(champion_path, "index", "index.bin")
    finalist=freeze_finalist(
        joinpath(sweep_path, "finalist_v3.toml"),
        checkpoint,
        index_file,
        partitions,
        manifest.preprocessing_hash;
        nprobe = nprobe,
        rare_threshold = rare_threshold,
    )
    index=SenBench.load_ivf_index(joinpath(champion_path, "index"))
    file_sha256(index_file)==String(finalist["index_sha256"])||error(
        "frozen finalist index changed",
    )
    println("running validation queries=$(length(validation)) confirmation=sealed")
    result=evaluate_hybrid_validation(
        dataset,
        index;
        nprobe = nprobe,
        k = k,
        repetitions = integer_env("SEN_VALIDATION_REPETITIONS", 3),
        rare_threshold = rare_threshold,
        target_recall = Float64(finalist["target_recall"]),
    )
    saved=save_hybrid_validation(joinpath(sweep_path, "validation_v3"), result)
    print_validation(result)
    println(
        "validation_recall_gate=$(result.recall_gate) speedup_p95=$(round(result.speedup_p95;digits=4,))",
    )
    println("summary=$(saved.summary)")
    return result
end

main()
