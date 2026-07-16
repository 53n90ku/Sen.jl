struct QualityWorkloadSpec
    id::String
    metric::Symbol
    filter_workload::Symbol
    vector_count::Int
    dimension::Int
    train_query_count::Int
    heldout_query_count::Int
    nlists::Int
    nprobes::Vector{Int}
    k::Int
    target_recall::Float64
    max_p95_ms::Float64
    probe_safety_factor::Float64
    repetitions::Int
    iterations::Int
    seed::Int
    selectivity::Float64
    methods::Vector{Symbol}
    dataset_sha256::String
    workload_sha256::String
end

struct QualityContract
    contract_version::VersionNumber
    index_version::Int
    generator_version::String
    workloads::Vector{QualityWorkloadSpec}
end

const QUALITY_METHODS=Set([:ivf,:ivf_prefilter,:ivf_postfilter,:filter_aware,:filter_aware_bound])

function quality_required(table::AbstractDict,key::String,context::String)
    haskey(table,key)||throw(ArgumentError("$context is missing $key"))
    return table[key]
end

function quality_workload_spec(table::AbstractDict)
    id=String(quality_required(table,"id","quality workload"))
    context="quality workload $(repr(id))"
    metric=Symbol(quality_required(table,"metric",context))
    filter_workload=Symbol(quality_required(table,"filter_workload",context))
    vector_count=Int(quality_required(table,"vector_count",context))
    dimension=Int(quality_required(table,"dimension",context))
    train_query_count=Int(quality_required(table,"train_query_count",context))
    heldout_query_count=Int(quality_required(table,"heldout_query_count",context))
    nlists=Int(quality_required(table,"nlists",context))
    nprobes=sort!(unique(Int.(quality_required(table,"nprobes",context))))
    k=Int(quality_required(table,"k",context))
    target_recall=Float64(quality_required(table,"target_recall",context))
    max_p95_ms=Float64(quality_required(table,"max_p95_ms",context))
    probe_safety_factor=Float64(quality_required(table,"probe_safety_factor",context))
    repetitions=Int(quality_required(table,"repetitions",context))
    iterations=Int(quality_required(table,"iterations",context))
    seed=Int(quality_required(table,"seed",context))
    selectivity=Float64(quality_required(table,"selectivity",context))
    methods=Symbol.(quality_required(table,"methods",context))
    dataset_sha256=String(quality_required(table,"dataset_sha256",context))
    workload_sha256=String(quality_required(table,"workload_sha256",context))

    metric in (:cosine,:dot)||throw(ArgumentError("$context has an unsupported metric"))
    filter_workload in (:none,:selected)||throw(ArgumentError("$context has an unsupported filter workload"))
    vector_count>0||throw(ArgumentError("$context vector count must be positive"))
    dimension>0||throw(ArgumentError("$context dimension must be positive"))
    train_query_count>0||throw(ArgumentError("$context train query count must be positive"))
    heldout_query_count>0||throw(ArgumentError("$context heldout query count must be positive"))
    1<=nlists<=vector_count||throw(ArgumentError("$context list count is invalid"))
    !isempty(nprobes)&&all(probe->1<=probe<=nlists,nprobes)||throw(ArgumentError("$context nprobes are invalid"))
    maximum(nprobes)==nlists||throw(ArgumentError("$context must include a full-probe recall point"))
    k>0||throw(ArgumentError("$context k must be positive"))
    0.0<target_recall<=1.0||throw(ArgumentError("$context target recall is invalid"))
    max_p95_ms>0||throw(ArgumentError("$context p95 ceiling must be positive"))
    probe_safety_factor>=1.0||throw(ArgumentError("$context probe safety factor must be at least one"))
    repetitions>0||throw(ArgumentError("$context repetitions must be positive"))
    iterations>0||throw(ArgumentError("$context iterations must be positive"))
    0.0<=selectivity<=1.0||throw(ArgumentError("$context selectivity is invalid"))
    !isempty(methods)&&Set(methods)⊆QUALITY_METHODS||throw(ArgumentError("$context methods are invalid"))
    length(methods)==length(Set(methods))||throw(ArgumentError("$context methods must be unique"))
    filter_workload===:none&&methods!=[:ivf]&&throw(ArgumentError("$context unfiltered workloads must measure IVF"))
    filter_workload===:selected&&(:ivf in methods)&&throw(ArgumentError("$context filtered workloads cannot use unfiltered IVF"))
    metric===:dot&&(:filter_aware_bound in methods)&&throw(ArgumentError("$context dot workloads cannot use the cosine bound strategy"))
    all(value->isempty(value)||occursin(r"^[0-9a-f]{64}$",value),(dataset_sha256,workload_sha256,))||throw(ArgumentError("$context fingerprints are invalid"))

    return QualityWorkloadSpec(id,metric,filter_workload,vector_count,dimension,train_query_count,heldout_query_count,nlists,nprobes,k,target_recall,max_p95_ms,probe_safety_factor,repetitions,iterations,seed,selectivity,methods,dataset_sha256,workload_sha256)
end

function load_quality_contract(path::AbstractString)
    isfile(path)||throw(ArgumentError("quality contract does not exist"))
    data=TOML.parsefile(path)
    get(data,"format_version",nothing)==1||throw(ArgumentError("unsupported quality contract format"))
    contract_version=VersionNumber(String(quality_required(data,"contract_version","quality contract")))
    index_version=Int(quality_required(data,"index_version","quality contract"))
    generator_version=String(quality_required(data,"generator_version","quality contract"))
    index_version==Sen.IVF_INDEX_VERSION||throw(ArgumentError("quality contract index version does not match Sen"))
    !isempty(generator_version)||throw(ArgumentError("quality contract generator version cannot be empty"))
    raw_workloads=quality_required(data,"workloads","quality contract")
    raw_workloads isa AbstractVector&&!isempty(raw_workloads)||throw(ArgumentError("quality contract workloads cannot be empty"))
    workloads=QualityWorkloadSpec[quality_workload_spec(table) for table in raw_workloads]
    length(workloads)==length(Set(spec.id for spec in workloads))||throw(ArgumentError("quality workload IDs must be unique"))
    return QualityContract(contract_version,index_version,generator_version,workloads)
end

function generate_quality_dataset(spec::QualityWorkloadSpec)
    dataset=generate_clustered_dataset(spec.vector_count,spec.dimension;clusters=spec.nlists,cluster_noise=0.12,cluster_skew=0.35,seed=spec.seed,)
    train=generate_clustered_queries(spec.train_query_count,dataset.centers;query_noise=0.12,cluster_skew=0.35,seed=spec.seed+1,).queries
    heldout=generate_clustered_queries(spec.heldout_query_count,dataset.centers;query_noise=0.12,cluster_skew=0.35,seed=spec.seed+2,).queries
    selected_clusters=clamp(round(Int,spec.nlists*spec.selectivity),1,spec.nlists)
    metadata=[(selected=dataset.assignments[index]<=selected_clusters,cluster=dataset.assignments[index],) for index in 1:spec.vector_count]
    filter=spec.filter_workload===:none ? nothing : (selected=true,)
    return(vectors=dataset.vectors,metadata=metadata,train_queries=train,heldout_queries=heldout,filter=filter,)
end

function quality_dataset_fingerprint(spec::QualityWorkloadSpec,data)
    io=IOBuffer()
    write(io,"sen-quality-dataset-v1\n")
    write(io,Int64(spec.vector_count),Int64(spec.dimension),Int64(spec.train_query_count),Int64(spec.heldout_query_count),Int64(spec.seed))
    write(io,data.vectors)
    write(io,data.train_queries)
    write(io,data.heldout_queries)

    for row in data.metadata
        write(io,row.selected ? UInt8(1) : UInt8(0))
        write(io,Int64(row.cluster))
    end

    return bytes2hex(SHA.sha256(take!(io)))
end

function quality_workload_fingerprint(contract::QualityContract,spec::QualityWorkloadSpec,dataset_sha256::AbstractString)
    value=join((
        "sen-quality-workload-v1",
        string(contract.contract_version),
        string(contract.index_version),
        contract.generator_version,
        spec.id,
        String(spec.metric),
        String(spec.filter_workload),
        string(spec.vector_count),
        string(spec.dimension),
        string(spec.train_query_count),
        string(spec.heldout_query_count),
        string(spec.nlists),
        join(spec.nprobes,','),
        string(spec.k),
        string(spec.target_recall),
        string(spec.max_p95_ms),
        string(spec.probe_safety_factor),
        string(spec.repetitions),
        string(spec.iterations),
        string(spec.seed),
        string(spec.selectivity),
        join(String.(spec.methods),','),
        String(dataset_sha256),
    ),'|')
    return bytes2hex(SHA.sha256(value))
end

function quality_search(db::VectorDB,query::AbstractVector,spec::QualityWorkloadSpec,method::Symbol,nprobe::Int,filter)
    if method===:ivf
        return search(db,query;k=spec.k,strategy=:ivf,nprobe=nprobe,)
    elseif method===:ivf_prefilter
        return search(db,query;k=spec.k,strategy=:prefilter,nprobe=nprobe,filter=filter,)
    elseif method===:ivf_postfilter
        return search(db,query;k=spec.k,strategy=:postfilter,nprobe=nprobe,filter=filter,adaptive_postfilter=false,postfilter_oversample=64,)
    elseif method===:filter_aware
        return search(db,query;k=spec.k,strategy=:filter_aware,nprobe=nprobe,max_nprobe=nprobe,filter=filter,adaptive=false,)
    elseif method===:filter_aware_bound
        return search(db,query;k=spec.k,strategy=:bound,nprobe=nprobe,max_nprobe=nprobe,filter=filter,)
    end

    throw(ArgumentError("unsupported quality method"))
end

function quality_truth(db::VectorDB,queries::AbstractMatrix,spec::QualityWorkloadSpec,filter)
    return[[result.id for result in search(db,@view(queries[:,index]);k=spec.k,strategy=:exact,filter=filter,)] for index in axes(queries,2)]
end

function quality_average_recall(db::VectorDB,queries::AbstractMatrix,truth,spec::QualityWorkloadSpec,method::Symbol,nprobe::Int,filter)
    total=0.0

    for index in axes(queries,2)
        predicted=Int[result.id for result in quality_search(db,@view(queries[:,index]),spec,method,nprobe,filter)]
        total+=recall_at_k(predicted,truth[index],spec.k)
    end

    return total/size(queries,2)
end

function select_quality_nprobe(db::VectorDB,data,spec::QualityWorkloadSpec,method::Symbol,truth)
    curve=[(nprobe=nprobe,recall=quality_average_recall(db,data.train_queries,truth,spec,method,nprobe,data.filter),) for nprobe in spec.nprobes]
    eligible=[point for point in curve if point.recall>=spec.target_recall]
    base=isempty(eligible) ? last(curve) : first(eligible)
    selected_target=ceil(Int,base.nprobe*spec.probe_safety_factor)
    candidates=[point for point in curve if point.nprobe>=selected_target]
    selected=isempty(candidates) ? last(curve) : first(candidates)
    return(curve=curve,selected_nprobe=selected.nprobe,train_recall=selected.recall,train_gate=base.recall>=spec.target_recall,)
end

function quality_latency(db::VectorDB,queries::AbstractMatrix,spec::QualityWorkloadSpec,method::Symbol,nprobe::Int,filter)
    times=Int[]

    for index in axes(queries,2)
        query=@view queries[:,index]
        quality_search(db,query,spec,method,nprobe,filter)

        for _ in 1:spec.repetitions
            started=time_ns()
            quality_search(db,query,spec,method,nprobe,filter)
            push!(times,time_ns()-started)
        end
    end

    return latency_summary(times)
end

function run_quality_workload(contract::QualityContract,spec::QualityWorkloadSpec;verify_fingerprints::Bool=true,)
    data=generate_quality_dataset(spec)
    dataset_sha256=quality_dataset_fingerprint(spec,data)
    workload_sha256=quality_workload_fingerprint(contract,spec,dataset_sha256)

    if verify_fingerprints
        dataset_sha256==spec.dataset_sha256||throw(ArgumentError("quality workload $(spec.id) dataset fingerprint changed"))
        workload_sha256==spec.workload_sha256||throw(ArgumentError("quality workload $(spec.id) contract fingerprint changed"))
    end

    db=create_db("quality-$(spec.id)";dim=spec.dimension,metric=spec.metric,durable=false,maintenance_config=MaintenanceConfig(enabled=false,),)

    try
        insert!(db,data.vectors,data.metadata;ids=collect(1:spec.vector_count),)
        build!(db;nlists=spec.nlists,iterations=spec.iterations,seed=spec.seed,restarts=1,training_count=spec.vector_count,)
        train_truth=quality_truth(db,data.train_queries,spec,data.filter)
        heldout_truth=quality_truth(db,data.heldout_queries,spec,data.filter)
        results=NamedTuple[]

        for method in spec.methods
            selection=select_quality_nprobe(db,data,spec,method,train_truth)
            heldout_recall=quality_average_recall(db,data.heldout_queries,heldout_truth,spec,method,selection.selected_nprobe,data.filter)
            latency=quality_latency(db,data.heldout_queries,spec,method,selection.selected_nprobe,data.filter)
            push!(results,(
                method=method,
                selected_nprobe=selection.selected_nprobe,
                train_recall=selection.train_recall,
                heldout_recall=heldout_recall,
                p50_ms=latency.p50_ms,
                p95_ms=latency.p95_ms,
                recall_passed=selection.train_gate&&heldout_recall>=spec.target_recall,
                latency_passed=latency.p95_ms<=spec.max_p95_ms,
            ))
        end

        return(id=spec.id,metric=spec.metric,filter_workload=spec.filter_workload,index_version=contract.index_version,dataset_sha256=dataset_sha256,workload_sha256=workload_sha256,target_recall=spec.target_recall,max_p95_ms=spec.max_p95_ms,methods=results,passed=all(result->result.recall_passed&&result.latency_passed,results),)
    finally
        close(db)
    end
end

function validate_quality_contract(path::AbstractString;run_benchmarks::Bool=true,verify_fingerprints::Bool=true,)
    contract=load_quality_contract(path)
    methods=Set(method for spec in contract.workloads for method in spec.methods)
    methods==QUALITY_METHODS||throw(ArgumentError("quality contract does not cover every approximate strategy"))
    Set(spec.metric for spec in contract.workloads)==Set([:cosine,:dot])||throw(ArgumentError("quality contract must cover cosine and dot metrics"))
    all(metric->any(spec->spec.metric===metric&&spec.filter_workload===:none,contract.workloads),(:cosine,:dot,))||throw(ArgumentError("quality contract must include unfiltered cosine and dot workloads"))
    run_benchmarks||return(contract=contract,results=NamedTuple[],passed=true,)
    results=[run_quality_workload(contract,spec;verify_fingerprints=verify_fingerprints,) for spec in contract.workloads]
    return(contract=contract,results=results,passed=all(result->result.passed,results),)
end
