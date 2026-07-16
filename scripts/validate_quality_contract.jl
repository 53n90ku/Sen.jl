using SenBench

root=normpath(joinpath(@__DIR__,".."))
contract_path=joinpath(root,"quality_contract.toml")
known=Set(["--fingerprints","--quiet"])
unknown=[argument for argument in ARGS if !(argument in known)]
isempty(unknown)||throw(ArgumentError("unknown arguments: $(join(unknown,", "))"))
fingerprints="--fingerprints" in ARGS
quiet="--quiet" in ARGS
report=validate_quality_contract(contract_path;run_benchmarks=!fingerprints,verify_fingerprints=!fingerprints,)

if fingerprints
    for spec in report.contract.workloads
        data=generate_quality_dataset(spec)
        dataset=quality_dataset_fingerprint(spec,data)
        workload=quality_workload_fingerprint(report.contract,spec,dataset)
        println("$(spec.id) dataset_sha256=$(dataset) workload_sha256=$(workload)")
    end
elseif !quiet
    println("Sen quality contract $(report.contract.contract_version), index version $(report.contract.index_version)")

    for workload in report.results
        println("[$(workload.passed ? "PASS" : "FAIL")] $(workload.id) dataset=$(workload.dataset_sha256) workload=$(workload.workload_sha256)")

        for method in workload.methods
            marker=method.recall_passed&&method.latency_passed ? "PASS" : "FAIL"
            println("  [$marker] $(method.method) nprobe=$(method.selected_nprobe) recall=$(round(method.heldout_recall;digits=4)) p95_ms=$(round(method.p95_ms;digits=3))")
        end
    end
end

report.passed||error("quality contract failed")
