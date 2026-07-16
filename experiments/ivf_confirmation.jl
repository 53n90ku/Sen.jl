pushfirst!(LOAD_PATH, normpath(joinpath(@__DIR__, "..", "benchmark", "SenBench")))

using SenBench
using TOML

function integer_env(name::String, default::Int)
    value=parse(Int, get(ENV, name, string(default)))
    value>0||error("$(name) must be positive")
    return value
end

function main()
    get(ENV, "SEN_CONFIRM", "0")=="1"||error(
        "confirmation is sealed; set SEN_CONFIRM=1 only after the finalist is frozen",
    )
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
    finalist_path=joinpath(sweep_path, "finalist_v3.toml")
    output_path=joinpath(sweep_path, "confirmation_v3")
    started_path=joinpath(output_path, "started.toml")
    complete_path=joinpath(output_path, "complete.toml")
    isfile(complete_path)&&error("confirmation already completed; refusing to rerun")
    isfile(started_path)&&error(
        "confirmation was already opened; inspect the existing run before doing anything else",
    )
    finalist=TOML.parsefile(finalist_path)
    seed=Int(finalist["seed"])
    manifest=SenBench.arxiv_manifest(data_path, scale, seed)
    String(finalist["dataset_hash"])==manifest.preprocessing_hash||error(
        "finalist dataset changed",
    )
    partitions=load_query_partitions(
        partition_path;
        dataset_hash = manifest.preprocessing_hash,
        allow_confirmation = true,
    )
    partitions.confirmation_hash==String(finalist["confirmation_hash"])||error(
        "sealed confirmation partition changed",
    )
    confirmation=query_partition_indices(
        partitions,
        :confirmation;
        allow_confirmation = true,
    )
    development=query_partition_indices(partitions, :development)
    dataset=load_arxiv_fanns_indices(
        data_path,
        development,
        confirmation;
        scale = scale,
        k = integer_env("SEN_REAL_K", 10),
        seed = seed,
    )
    index_file=joinpath(champion_path, "index", "index.bin")
    file_sha256(index_file)==String(finalist["index_sha256"])||error(
        "frozen finalist index changed",
    )
    index=SenBench.load_ivf_index(joinpath(champion_path, "index"))
    mkpath(output_path)
    open(started_path, "w") do io
        TOML.print(
            io,
            Dict(
                "configuration_hash"=>String(finalist["configuration_hash"]),
                "confirmation_hash"=>partitions.confirmation_hash,
                "query_count"=>length(confirmation),
                "started_unix"=>time(),
            ),
        )
    end
    println("opening sealed confirmation queries=$(length(confirmation))")
    result=evaluate_hybrid_validation(
        dataset,
        index;
        nprobe = Int(finalist["nprobe"]),
        k = integer_env("SEN_REAL_K", 10),
        repetitions = integer_env("SEN_CONFIRMATION_REPETITIONS", 3),
        rare_threshold = Float64(finalist["rare_threshold"]),
        target_recall = Float64(finalist["target_recall"]),
        split = :confirmation,
    )
    saved=save_hybrid_validation(output_path, result)
    speed_gate=result.speedup_p95>=1.25
    bucket_gate=all(
        summary->summary.average_recall>=Float64(finalist["target_recall"]),
        values(result.buckets),
    )
    passed=result.recall_gate&&speed_gate&&bucket_gate
    complete=Dict(
        "passed"=>passed,
        "recall_gate"=>result.recall_gate,
        "speed_gate"=>speed_gate,
        "bucket_gate"=>bucket_gate,
        "recall"=>result.hybrid.summary.average_recall,
        "recall_ci_lower"=>result.recall_interval.lower,
        "recall_ci_upper"=>result.recall_interval.upper,
        "exact_p95_ms"=>result.exact.summary.latency.p95_ms,
        "hybrid_p95_ms"=>result.hybrid.summary.latency.p95_ms,
        "speedup_p95"=>result.speedup_p95,
        "average_candidates_scored"=>result.hybrid.summary.average_candidates_scored,
        "completed_unix"=>time(),
    )
    temporary="$(complete_path).tmp"
    open(temporary, "w") do io
        TOML.print(io, complete)
    end
    mv(temporary, complete_path; force = true)
    println(
        "confirmation recall=$(round(complete["recall"];digits=4,)) ci95=[$(round(complete["recall_ci_lower"];digits=4,)),$(round(complete["recall_ci_upper"];digits=4,))]",
    )
    println(
        "confirmation exact_p95_ms=$(round(complete["exact_p95_ms"];digits=4,)) hybrid_p95_ms=$(round(complete["hybrid_p95_ms"];digits=4,)) speedup=$(round(complete["speedup_p95"];digits=4,))",
    )
    println("confirmation passed=$(passed) summary=$(saved.summary)")
    return result
end

main()
