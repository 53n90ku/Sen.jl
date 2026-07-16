using Dates
using SenBench

root=normpath(joinpath(@__DIR__, ".."))
contract_path=joinpath(root, "quality_contract.toml")
output_path=joinpath(root, "BENCHMARKS.md")

function warm_release_benchmark(contract)
    run_quality_workload(contract, first(contract.workloads))
    return nothing
end

function release_benchmark_markdown(report)
    io=IOBuffer()
    cpu=isempty(Sys.cpu_info()) ? "unknown" : Sys.cpu_info()[1].model
    println(io, "# Sen v0.1 benchmark snapshot")
    println(io)
    println(
        io,
        "Generated $(Dates.format(now(), "yyyy-mm-dd HH:MM")) with Julia $(VERSION) on $(Sys.KERNEL) $(Sys.ARCH), $(Threads.nthreads()) Julia thread(s), CPU: $(cpu).",
    )
    println(io)
    println(
        io,
        "All rows use the frozen `quality_contract.toml` datasets, held-out queries, index format v$(report.contract.index_version), target recall 0.85, and a p95 latency ceiling of 10 ms.",
    )
    println(io)
    println(
        io,
        "| Workload | Metric | Filter | Method | Vectors | Dim | nprobe | Recall@10 | p95 ms | Build ms | Index KiB |",
    )
    println(io, "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")

    for workload in report.results
        for method in workload.methods
            println(
                io,
                "| $(workload.id) | $(workload.metric) | $(workload.filter_workload) | $(method.method) | $(workload.vector_count) | $(workload.dimension) | $(method.selected_nprobe) | $(round(method.heldout_recall; digits=4)) | $(round(method.p95_ms; digits=3)) | $(round(workload.build_ms; digits=2)) | $(round(workload.index_bytes/1024; digits=2)) |",
            )
        end
    end

    println(io)
    println(io, "Reproduce the report from the repository root:")
    println(io)
    println(io, "```bash")
    println(io, "julia --project=benchmark/SenBench scripts/benchmark_release.jl")
    println(io, "```")
    println(io)
    println(
        io,
        "Build timings follow one warm-up workload; query timings warm each query before measurement. Build and latency measurements are machine-dependent. Recall and dataset fingerprints are deterministic release gates; this snapshot is evidence for Sen on the machine shown above, not a cross-system performance claim.",
    )
    return String(take!(io))
end

contract=load_quality_contract(contract_path)
warm_release_benchmark(contract)
report=validate_quality_contract(contract_path)
report.passed||error("quality contract failed")
markdown=release_benchmark_markdown(report)
write(output_path, markdown)
print(markdown)
