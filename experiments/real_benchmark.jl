pushfirst!(LOAD_PATH,normpath(joinpath(@__DIR__,"..","benchmark","SenBench")))

using Sen
using SenBench

function integer_env(name::String,default::Int)
    value=parse(Int,get(ENV,name,string(default)))
    value>0||error("$(name) must be positive")
    return value
end

function real_profile(scale::Symbol)
    if scale===:small
        return(
            train_count=integer_env("SEN_REAL_TRAIN_QUERIES",12),
            heldout_count=integer_env("SEN_REAL_HELDOUT_QUERIES",24),
            nlists=integer_env("SEN_REAL_NLISTS",16),
            nprobes=[1,2,4,8,16],
            iterations=integer_env("SEN_REAL_ITERATIONS",5),
            restarts=integer_env("SEN_REAL_RESTARTS",1),
            training_count=1_000,
        )
    elseif scale===:medium
        return(
            train_count=integer_env("SEN_REAL_TRAIN_QUERIES",24),
            heldout_count=integer_env("SEN_REAL_HELDOUT_QUERIES",48),
            nlists=integer_env("SEN_REAL_NLISTS",64),
            nprobes=[1,2,4,8,16,32,64],
            iterations=integer_env("SEN_REAL_ITERATIONS",8),
            restarts=integer_env("SEN_REAL_RESTARTS",2),
            training_count=integer_env("SEN_REAL_TRAINING_COUNT",20_000),
        )
    end

    error("real benchmark supports small or medium scale on this machine")
end

function print_summary(result)
    println("method\trecall\tp50_ms\tp95_ms\tqps\tcandidates")
    for row in real_benchmark_summary_rows(result)
        println("$(row.method)\t$(round(row.heldout_recall;digits=4,))\t$(round(row.p50_ms;digits=4,))\t$(round(row.p95_ms;digits=4,))\t$(round(row.qps;digits=2,))\t$(round(row.candidates_scored;digits=2,))")
    end
end

function main()
    action=isempty(ARGS) ? "run" : ARGS[1]
    scale=Symbol(get(ENV,"SEN_REAL_SCALE",length(ARGS)>=2 ? ARGS[2] : "medium"))
    profile=real_profile(scale)
    root=normpath(joinpath(@__DIR__,".."))
    data_path=get(ENV,"SEN_REAL_DATA",joinpath(root,"data","arxiv-for-fanns-$(scale)"))
    output_path=get(ENV,"SEN_OUTPUT",joinpath(root,"results","real-$(scale)-$(round(Int,time()))"))
    seed=integer_env("SEN_REAL_SEED",42)
    k=integer_env("SEN_REAL_K",10)

    if action in ("download","prepare","run")
        download_arxiv_fanns(data_path;scale=scale,)
    else
        error("action must be download, prepare or run")
    end

    action=="download"&&return nothing
    println("loading dataset scale=$(scale)")
    dataset=load_arxiv_fanns(
        data_path;
        scale=scale,
        train_count=profile.train_count,
        heldout_count=profile.heldout_count,
        k=k,
        seed=seed,
        verify_hashes=get(ENV,"SEN_VERIFY_HASHES","0")=="1",
    )
    verification_count=integer_env("SEN_REAL_VERIFY_QUERIES",1)
    println("verifying published groundtruth queries=$(verification_count)")
    verification=verify_arxiv_groundtruth(dataset;sample_count=verification_count,k=k,)
    verification.passed||error("published groundtruth verification failed recalls=$(verification.recalls)")
    action=="prepare"&&return dataset

    println("running real benchmark vectors=$(size(dataset.vectors,2)) dimension=$(size(dataset.vectors,1))")
    result=run_real_benchmark(
        dataset;
        name="arxiv-fanns-$(scale)",
        nlists=profile.nlists,
        nprobes=profile.nprobes,
        k=k,
        target_recall=parse(Float64,get(ENV,"SEN_REAL_TARGET_RECALL","0.95")),
        repetitions=integer_env("SEN_REAL_REPETITIONS",2),
        iterations=profile.iterations,
        restarts=profile.restarts,
        training_count=profile.training_count,
        seed=seed,
    )
    saved=save_real_benchmark(output_path,result)
    print_summary(result)
    println("recall_gate=$(result.recall_gate) planner_gate=$(result.planner_gate) passed=$(result.passed)")
    println("oracle=$(result.oracle_method) planner_regret=$(round(result.planner_regret;digits=4,))")
    println("summary=$(saved.summary)")
    println("environment=$(saved.environment)")
    return result
end

main()
