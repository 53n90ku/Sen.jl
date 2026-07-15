pushfirst!(LOAD_PATH,normpath(joinpath(@__DIR__,"..","benchmark","SenBench")))

using Sen
using Sen: build_ivf
using SenBench

function experiment_name(profile::String,vector_count::Int,dimension::Int,vector_workload::Symbol,filter_workload::Symbol,selectivity::Float64,seed::Int)
    selectivity_name=replace(string(selectivity),'.'=>'_')
    return "$(profile)-n$(vector_count)-d$(dimension)-$(vector_workload)-$(filter_workload)-s$(selectivity_name)-seed$(seed)"
end

function benchmark_spec(profile::String,vector_count::Int,dimension::Int,vector_workload::Symbol,filter_workload::Symbol,selectivity::Float64,seed::Int;train_queries::Int,heldout_queries::Int,nlists::Int,iterations::Int,restarts::Int,repetitions::Int,probe_safety_factor::Float64=8.0,postfilter_safety_factor::Float64=2.0,candidate_multiplier::Float64=16.0,vector_weight::Float64=0.5,filter_weight::Float64=0.5,)
    nprobes=unique(sort([1,2,4,8,16,32,64,nlists]))
    filter!(nprobe->nprobe<=nlists,nprobes)
    return ExperimentSpec(
        experiment_name(profile,vector_count,dimension,vector_workload,filter_workload,selectivity,seed);
        vector_count=vector_count,
        dimension=dimension,
        train_query_count=train_queries,
        heldout_query_count=heldout_queries,
        nlists=nlists,
        nprobes=nprobes,
        k=10,
        target_recall=0.95,
        selection_margin=0.02,
        probe_safety_factor=probe_safety_factor,
        minimum_speedup=1.15,
        recall_tolerance=0.01,
        candidate_multiplier=candidate_multiplier,
        postfilter_safety_factor=postfilter_safety_factor,
        vector_weight=vector_weight,
        filter_weight=filter_weight,
        repetitions=repetitions,
        iterations=iterations,
        restarts=restarts,
        training_count=min(vector_count,20_000),
        vector_workload=vector_workload,
        filter_workload=filter_workload,
        selectivity=selectivity,
        seed=seed,
    )
end

function profile_specs(profile::String)
    specs=ExperimentSpec[]

    if profile=="pilot"
        push!(specs,benchmark_spec(profile,2_000,32,:clustered,:correlated,0.05,42;train_queries=10,heldout_queries=20,nlists=16,iterations=5,restarts=1,repetitions=2,))
    elseif profile=="claim"
        for seed in (42,43,44)
            for selectivity in (0.01,0.05,0.10)
                push!(specs,benchmark_spec(profile,10_000,128,:clustered,:correlated,selectivity,seed;train_queries=30,heldout_queries=100,nlists=64,iterations=8,restarts=2,repetitions=3,))
            end
        end
    elseif profile=="claim100k"
        for seed in (42,43,44)
            for selectivity in (0.01,0.05,0.10)
                push!(specs,benchmark_spec(profile,100_000,128,:clustered,:correlated,selectivity,seed;train_queries=30,heldout_queries=100,nlists=64,iterations=6,restarts=2,repetitions=3,))
            end
        end
    elseif profile=="confirm100k"
        for seed in (45,46,47)
            for selectivity in (0.01,0.05,0.10)
                push!(specs,benchmark_spec(profile,100_000,128,:clustered,:correlated,selectivity,seed;train_queries=30,heldout_queries=100,nlists=64,iterations=6,restarts=2,repetitions=3,))
            end
        end
    elseif profile=="final100k"
        for seed in (48,49,50)
            for selectivity in (0.01,0.05,0.10)
                push!(specs,benchmark_spec(profile,100_000,128,:clustered,:correlated,selectivity,seed;train_queries=100,heldout_queries=200,nlists=64,iterations=6,restarts=2,repetitions=3,))
            end
        end
    elseif profile=="boundary"
        for filter_workload in (:random,:correlated,:anticorrelated,:skewed)
            for selectivity in (0.001,0.01,0.05,0.10,0.50,1.0)
                push!(specs,benchmark_spec(profile,10_000,128,:clustered,filter_workload,selectivity,51;train_queries=50,heldout_queries=100,nlists=64,iterations=8,restarts=2,repetitions=3,))
            end
        end
    elseif profile=="dimension100k"
        for dimension in (128,384,768)
            push!(specs,benchmark_spec(profile,100_000,dimension,:clustered,:correlated,0.05,53;train_queries=30,heldout_queries=50,nlists=64,iterations=6,restarts=2,repetitions=2,))
        end
    elseif profile=="scale1m"
        for selectivity in (0.01,0.05,0.10)
            push!(specs,benchmark_spec(profile,1_000_000,128,:clustered,:correlated,selectivity,52;train_queries=20,heldout_queries=30,nlists=64,iterations=6,restarts=2,repetitions=2,))
        end
    elseif profile=="discovery"
        for vector_workload in (:gaussian,:clustered)
            for filter_workload in (:random,:correlated,:anticorrelated,:skewed)
                for selectivity in (0.001,0.01,0.05,0.10,0.50,1.0)
                    push!(specs,benchmark_spec(profile,10_000,128,vector_workload,filter_workload,selectivity,42;train_queries=20,heldout_queries=50,nlists=64,iterations=8,restarts=2,repetitions=2,))
                end
            end
        end
    elseif profile=="confirmation"
        for dimension in (128,384)
            for vector_workload in (:gaussian,:clustered)
                for seed in (42,43,44)
                    for filter_workload in (:random,:correlated,:anticorrelated,:skewed)
                        for selectivity in (0.01,0.05,0.10)
                            push!(specs,benchmark_spec(profile,100_000,dimension,vector_workload,filter_workload,selectivity,seed;train_queries=30,heldout_queries=100,nlists=64,iterations=8,restarts=2,repetitions=3,))
                        end
                    end
                end
            end
        end
    elseif profile=="scale"
        for vector_workload in (:gaussian,:clustered)
            for filter_workload in (:random,:correlated,:anticorrelated,:skewed)
                for selectivity in (0.01,0.05)
                    push!(specs,benchmark_spec(profile,1_000_000,128,vector_workload,filter_workload,selectivity,42;train_queries=10,heldout_queries=20,nlists=64,iterations=6,restarts=2,repetitions=1,))
                end
            end
        end
    else
        error("profile must be pilot, claim, claim100k, confirm100k, final100k, boundary, dimension100k, scale1m, discovery, confirmation or scale")
    end

    maximum_count=parse(Int,get(ENV,"SEN_MAX_EXPERIMENTS",string(length(specs))))
    maximum_count>0||error("SEN_MAX_EXPERIMENTS must be positive")
    return specs[1:min(maximum_count,length(specs))]
end

function main()
    profile=get(ENV,"SEN_BENCH_PROFILE",isempty(ARGS) ? "pilot" : ARGS[1])
    output_path=get(ENV,"SEN_OUTPUT",joinpath(@__DIR__,"..","results","$(profile)-$(round(Int,time()))"))
    specs=profile_specs(profile)
    results=Any[]

    println("profile=$(profile) experiments=$(length(specs)) output=$(output_path)")
    println("experiment\toracle\tplanner_regret\tclaim\tspeedup_p95")

    active_key=nothing
    active_base=nothing
    active_ivf=nothing
    active_postfilter_rankings=nothing
    active_postfilter_ranking_seconds=0.0
    active_build_seconds=0.0

    for spec in specs
        key=(spec.vector_count,spec.dimension,spec.train_query_count,spec.heldout_query_count,spec.nlists,spec.iterations,spec.restarts,spec.training_count,spec.metric,spec.vector_workload,spec.seed,)

        if active_key!=key
            active_base=generate_experiment_base(spec)
            active_ivf=nothing
            active_build_seconds=@elapsed active_ivf=build_ivf(active_base.vectors;nlists=spec.nlists,iterations=spec.iterations,seed=spec.seed,metric=spec.metric,restarts=spec.restarts,training_count=spec.training_count,)
            active_postfilter_ranking_seconds=@elapsed active_postfilter_rankings=build_postfilter_rankings(spec,active_ivf,active_base.vectors,active_base.train_queries)
            active_key=key
        end

        result=run_claim_benchmark(spec;base=active_base,prebuilt_ivf=active_ivf,prebuilt_build_seconds=active_build_seconds,prebuilt_postfilter_rankings=active_postfilter_rankings,prebuilt_postfilter_ranking_seconds=active_postfilter_ranking_seconds,)
        push!(results,result)
        println("$(spec.name)\t$(result.oracle_method)\t$(round(result.planner_regret;digits=4,))\t$(result.claim.passed)\t$(round(result.claim.speedup_p95;digits=4,))")
        GC.gc()
    end

    saved=save_experiment_suite(output_path,results)
    println("raw=$(saved.raw)")
    println("summary=$(saved.summary)")
    println("aggregate=$(saved.aggregate)")
    println("claims=$(saved.claims)")
    println("environment=$(saved.environment)")

    if get(ENV,"SEN_REQUIRE_CLAIM","0")=="1"
        all(result->result.claim.passed,results)||error("one or more claim gates failed")
    end

    return results
end

main()
