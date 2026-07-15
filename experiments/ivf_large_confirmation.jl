pushfirst!(LOAD_PATH,normpath(joinpath(@__DIR__,"..","benchmark","SenBench")))

using SenBench
using TOML

function integer_env(name::String,default::Int)
    value=parse(Int,get(ENV,name,string(default)))
    value>0||error("$(name) must be positive")
    return value
end

function finalist_candidate(path::String)
    configured=get(ENV,"SEN_FINALIST_INDEX","")
    !isempty(configured)&&return configured
    candidates=String[]
    for name in readdir(path)
        occursin("lists-1536-",name)||continue
        occursin("-iter-8-restart-2-",name)||continue
        candidate=joinpath(path,name)
        isfile(joinpath(candidate,"validation","summary.tsv"))&&push!(candidates,candidate)
    end
    length(candidates)==1||error("expected exactly one default finalist")
    return only(candidates)
end

function freeze_finalist(path::String,candidate::String,manifest::DatasetManifest,partitions::RealQueryPartitions)
    checkpoint=TOML.parsefile(joinpath(candidate,"checkpoint.toml"))
    index_file=joinpath(candidate,"index","index.bin")
    data=Dict(
        "dataset_hash"=>manifest.preprocessing_hash,
        "partition_seed"=>partitions.seed,
        "confirmation_hash"=>partitions.confirmation_hash,
        "configuration_hash"=>String(checkpoint["configuration_hash"]),
        "candidate_path"=>candidate,
        "index_sha256"=>file_sha256(index_file),
        "nlists"=>Int(checkpoint["nlists"]),
        "training_count"=>Int(checkpoint["training_count"]),
        "iterations"=>Int(checkpoint["iterations"]),
        "restarts"=>Int(checkpoint["restarts"]),
        "seed"=>Int(checkpoint["seed"]),
        "nprobe"=>Int(checkpoint["selected_nprobe"]),
        "rare_threshold"=>parse(Float64,get(ENV,"SEN_RARE_THRESHOLD","0.01")),
        "target_recall"=>parse(Float64,get(ENV,"SEN_TARGET_RECALL","0.95")),
    )
    if isfile(path)
        TOML.parsefile(path)==data||error("frozen finalist changed")
        return data
    end
    temporary="$(path).tmp"
    open(temporary,"w") do io
        TOML.print(io,data)
    end
    mv(temporary,path;force=true,)
    println("frozen finalist path=$(path)")
    return data
end

function print_confirmation(result)
    summary=result.hybrid.summary
    interval=result.recall_interval
    println("confirmation recall=$(round(summary.average_recall;digits=6,)) ci=[$(round(interval.lower;digits=6,)),$(round(interval.upper;digits=6,))] p95_ms=$(round(summary.latency.p95_ms;digits=4,)) speedup=$(round(result.speedup_p95;digits=4,)) candidates=$(round(summary.average_candidates_scored;digits=2,)) gate=$(result.recall_gate)")
    for(bucket,bucket_summary) in sort!(collect(result.buckets);by=value->string(first(value)),)
        println("confirmation bucket=$(bucket) recall=$(round(bucket_summary.average_recall;digits=6,))")
    end
end

function main()
    root=normpath(joinpath(@__DIR__,".."))
    data_path=get(ENV,"SEN_REAL_DATA",joinpath(root,"data","arxiv-for-fanns-300k"))
    output_path=get(ENV,"SEN_SWEEP_OUTPUT",joinpath(root,"results","ivf-quality-300k"))
    partition_path=get(ENV,"SEN_QUERY_PARTITIONS",joinpath(data_path,"sen_query_partitions_300k_v1"))
    manifest=load_dataset_manifest(joinpath(data_path,"sen_dataset.toml"))
    public_partitions=load_query_partitions(partition_path;dataset_hash=manifest.preprocessing_hash,)
    candidate=finalist_candidate(output_path)
    frozen_path=joinpath(output_path,"finalist_v1.toml")
    frozen=freeze_finalist(frozen_path,candidate,manifest,public_partitions)
    confirmation_path=joinpath(output_path,"confirmation_v1")
    isfile(joinpath(confirmation_path,"summary.tsv"))&&error("confirmation already completed")
    partitions=load_query_partitions(partition_path;dataset_hash=manifest.preprocessing_hash,allow_confirmation=true,)
    partitions.confirmation_hash==String(frozen["confirmation_hash"])||error("sealed confirmation hash changed")
    development=query_partition_indices(partitions,:development)
    confirmation=query_partition_indices(partitions,:confirmation;allow_confirmation=true,)
    println("opening sealed confirmation queries=$(length(confirmation)) frozen=$(basename(candidate))")
    dataset=load_arxiv_subset(data_path,development,confirmation;k=integer_env("SEN_REAL_K",10),)
    index_file=joinpath(candidate,"index","index.bin")
    file_sha256(index_file)==String(frozen["index_sha256"])||error("frozen finalist index changed")
    index=SenBench.load_ivf_index(joinpath(candidate,"index"))
    result=evaluate_hybrid_validation(
        dataset,
        index;
        nprobe=Int(frozen["nprobe"]),
        k=integer_env("SEN_REAL_K",10),
        repetitions=integer_env("SEN_CONFIRMATION_REPETITIONS",5),
        rare_threshold=Float64(frozen["rare_threshold"]),
        target_recall=Float64(frozen["target_recall"]),
        split=:confirmation,
    )
    save_hybrid_validation(confirmation_path,result)
    print_confirmation(result)
    return result
end

main()
