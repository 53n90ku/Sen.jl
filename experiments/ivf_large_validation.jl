pushfirst!(LOAD_PATH, normpath(joinpath(@__DIR__, "..", "benchmark", "SenBench")))

using SenBench
using TOML

function integer_env(name::String, default::Int)
    value=parse(Int, get(ENV, name, string(default)))
    value>0||error("$(name) must be positive")
    return value
end

function refined_candidates(path::String)
    candidates=String[]
    for name in readdir(path)
        candidate=joinpath(path, name)
        occursin("-iter-8-restart-2-", name)||continue
        isfile(joinpath(candidate, "checkpoint.toml"))||continue
        isfile(joinpath(candidate, "index", "index.bin"))||continue
        push!(candidates, candidate)
    end
    isempty(candidates)&&error("no refined IVF candidates found")
    return sort!(candidates)
end

function print_result(name::String, result)
    summary=result.hybrid.summary
    println(
        "$(name) recall=$(round(summary.average_recall;digits=6,)) p95_ms=$(round(summary.latency.p95_ms;digits=4,)) speedup=$(round(result.speedup_p95;digits=4,)) candidates=$(round(summary.average_candidates_scored;digits=2,)) gate=$(result.recall_gate)",
    )
    for (bucket, bucket_summary) in
        sort!(collect(result.buckets); by = value->string(first(value)))
        println(
            "$(name) bucket=$(bucket) recall=$(round(bucket_summary.average_recall;digits=6,))",
        )
    end
end

function main()
    root=normpath(joinpath(@__DIR__, ".."))
    data_path=get(ENV, "SEN_REAL_DATA", joinpath(root, "data", "arxiv-for-fanns-300k"))
    output_path=get(ENV, "SEN_SWEEP_OUTPUT", joinpath(root, "results", "ivf-quality-300k"))
    partition_path=get(
        ENV,
        "SEN_QUERY_PARTITIONS",
        joinpath(data_path, "sen_query_partitions_300k_v1"),
    )
    manifest=load_dataset_manifest(joinpath(data_path, "sen_dataset.toml"))
    partitions=load_query_partitions(
        partition_path;
        dataset_hash = manifest.preprocessing_hash,
    )
    development=query_partition_indices(partitions, :development)
    validation=query_partition_indices(partitions, :validation)
    dataset=load_arxiv_subset(
        data_path,
        development,
        validation;
        k = integer_env("SEN_REAL_K", 10),
    )
    results=NamedTuple[]

    for candidate in refined_candidates(output_path)
        checkpoint=TOML.parsefile(joinpath(candidate, "checkpoint.toml"))
        index=SenBench.load_ivf_index(joinpath(candidate, "index"))
        nprobe=Int(checkpoint["selected_nprobe"])
        name=basename(candidate)
        println("validating candidate=$(name) nprobe=$(nprobe) confirmation=sealed")
        result=evaluate_hybrid_validation(
            dataset,
            index;
            nprobe = nprobe,
            k = integer_env("SEN_REAL_K", 10),
            repetitions = integer_env("SEN_VALIDATION_REPETITIONS", 3),
            rare_threshold = parse(Float64, get(ENV, "SEN_RARE_THRESHOLD", "0.01")),
            target_recall = parse(Float64, get(ENV, "SEN_TARGET_RECALL", "0.95")),
        )
        save_hybrid_validation(joinpath(candidate, "validation"), result)
        print_result(name, result)
        push!(results, (path = candidate, checkpoint = checkpoint, result = result))
    end
    return results
end

main()
