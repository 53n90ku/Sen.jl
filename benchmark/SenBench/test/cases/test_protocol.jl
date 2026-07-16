using Test
using TOML
using Sen

@testset "benchmark protocol" begin
    spec=ExperimentSpec(
        "test-protocol";
        vector_count = 100,
        dimension = 4,
        train_query_count = 3,
        heldout_query_count = 4,
        nlists = 4,
        nprobes = [4, 1, 2, 2],
        k = 3,
        target_recall = 0.50,
        selection_margin = 0.0,
        minimum_speedup = 1.01,
        repetitions = 1,
        iterations = 3,
        restarts = 1,
        training_count = 100,
        selectivity = 0.50,
        seed = 42,
    )

    @test spec.nprobes==[1, 2, 4]
    @test spec.probe_safety_factor==2.0
    @test spec.postfilter_multipliers==[4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
    @test spec.postfilter_safety_factor==2.0
    @test experiment_methods(spec)==[
        :exact,
        :ivf_prefilter,
        :ivf_postfilter,
        :filter_aware,
        :filter_aware_bound,
    ]
    @test_throws ArgumentError ExperimentSpec(
        "invalid";
        vector_count = 10,
        dimension = 2,
        nlists = 2,
        nprobes = Int[],
    )
    @test_throws ArgumentError ExperimentSpec(
        "invalid";
        vector_count = 10,
        dimension = 2,
        nlists = 2,
        nprobes = [3],
    )
    scale_spec=ExperimentSpec(
        "scale-grid";
        vector_count = 1_000_000,
        dimension = 1,
        nlists = 1,
        nprobes = [1],
        selectivity = 0.05,
    )
    scale_multipliers=SenBench.postfilter_multiplier_values(scale_spec)
    @test 256.0 in scale_multipliers
    @test 4096.0 in scale_multipliers
    @test 5000.0 in scale_multipliers

    base=generate_experiment_base(spec)
    ivf=build_ivf(
        base.vectors;
        nlists = spec.nlists,
        iterations = spec.iterations,
        seed = spec.seed,
        restarts = spec.restarts,
        training_count = spec.training_count,
    )
    postfilter_rankings=build_postfilter_rankings(
        spec,
        ivf,
        base.vectors,
        base.train_queries,
    )
    @test postfilter_rankings.nprobes==spec.nprobes
    @test length(postfilter_rankings.rankings)==spec.train_query_count
    ranking_metadata=[(position = index,) for index = 1:spec.vector_count]

    for nprobe in spec.nprobes
        query=@view base.train_queries[:, 1]
        candidate_count=count_ivf_candidates(ivf, query; nprobe = nprobe)
        expected=search_ivf(
            ivf,
            base.vectors,
            ranking_metadata,
            query;
            k = candidate_count,
            nprobe = nprobe,
        )
        @test postfilter_rankings.rankings[1][nprobe]==[result.index for result in expected]
    end

    result=run_claim_benchmark(
        spec;
        base = base,
        prebuilt_ivf = ivf,
        prebuilt_build_seconds = 0.01,
        prebuilt_postfilter_rankings = postfilter_rankings,
    )
    method_count=length(experiment_methods(spec))+1

    @test Set(keys(result.method_results))==Set(vcat(experiment_methods(spec), [:auto]))
    @test length(result.raw_rows)==method_count*spec.heldout_query_count*spec.repetitions
    @test result.oracle_method in experiment_methods(spec)
    @test result.planner_regret>=0
    @test result.method_results[:exact].summary.average_recall==1.0
    @test result.postfilter_candidate_multiplier>0
    @test result.postfilter_ranking_megabytes>0
    @test all(
        row->row.postfilter_candidate_multiplier==result.postfilter_candidate_multiplier,
        result.method_results[:ivf_postfilter].rows,
    )
    @test all(row->row.split=="heldout", result.raw_rows)
    summary_rows=experiment_summary_rows(result)
    @test length(summary_rows)==method_count
    @test only(
        row.postfilter_candidate_multiplier for
        row in summary_rows if row.method=="ivf_postfilter"
    )==result.postfilter_candidate_multiplier

    mktempdir() do path
        saved=save_experiment_suite(path, [result])
        @test isfile(saved.raw)
        @test isfile(saved.summary)
        @test isfile(saved.aggregate)
        @test isfile(saved.claims)
        @test isfile(saved.environment)
        @test startswith(read(saved.raw, String), "experiment\tsplit")
        environment=TOML.parsefile(saved.environment)
        @test environment["julia_threads"]>=1
        @test environment["experiments"][1]["name"]==spec.name
        @test environment["experiments"][1]["selected_postfilter_candidate_multiplier"]==result.postfilter_candidate_multiplier
    end
end

@testset "frozen quality contract" begin
    contract_path=joinpath(root, "quality_contract.toml")
    contract=load_quality_contract(contract_path)
    report=validate_quality_contract(contract_path; run_benchmarks = false)

    @test report.passed
    @test contract.index_version==Sen.IVF_INDEX_VERSION
    @test Set(spec.metric for spec in contract.workloads)==Set([:cosine, :dot])
    @test Set(method for spec in contract.workloads for method in spec.methods)==SenBench.QUALITY_METHODS
    @test all(
        metric->any(
            spec->spec.metric===metric&&spec.filter_workload===:none,
            contract.workloads,
        ),
        (:cosine, :dot),
    )

    for spec in contract.workloads
        data=generate_quality_dataset(spec)
        dataset_sha256=quality_dataset_fingerprint(spec, data)
        @test dataset_sha256==spec.dataset_sha256
        @test quality_workload_fingerprint(contract, spec, dataset_sha256)==spec.workload_sha256
    end

    mktempdir() do path
        invalid=TOML.parsefile(contract_path)
        invalid["index_version"]+=1
        invalid_path=joinpath(path, "quality_contract.toml")
        open(invalid_path, "w") do io
            TOML.print(io, invalid)
        end
        @test_throws ArgumentError load_quality_contract(invalid_path)
    end
end
