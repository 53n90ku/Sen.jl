using Test
using Random
using Sen

@testset "synthetic dataset" begin
    dataset = generate_synthetic_dataset(100,16;seed = 42)

    @test size(dataset.vectors)==(16,100)
    @test length(dataset.metadata)==100
    
    for i in 1:100
        vector = @view dataset.vectors[:,i]
        @test isapprox(sum(abs2,vector),1.0f0;atol = 1.0f-5)
    end

    @test dataset.metadata[1].topic in (
        "systems","machine-learning","databases",
    )
end

@testset "batch search benchmark" begin
    rng=MersenneTwister(94)
    db=create_db("batch-benchmark";dim=4,durable=false,)
    vectors=randn(rng,Float32,4,40)
    metadata=[(group=index%2,) for index in 1:40]
    queries=randn(rng,Float32,4,8)

    insert!(db,vectors,metadata;ids=collect(1:40),)
    result=run_batch_search_benchmark(db,queries;repetitions=2,workers=2,k=4,strategy=:exact,parallel_threshold=2,)

    @test result.query_count==8
    @test result.workers==min(2,Threads.nthreads(:default))
    @test result.serial.latency.p95_ms>=0
    @test result.parallel.latency.p95_ms>=0
    @test result.serial.queries_per_second>0
    @test result.parallel.queries_per_second>0
    @test result.throughput_speedup>0
    @test batch_queries_per_second(10,1000.0)==10.0
    @test_throws ArgumentError batch_queries_per_second(0,1.0)
    @test_throws ArgumentError run_batch_search_benchmark(db,zeros(Float32,4,0);repetitions=1,)
end

@testset "vector load benchmark" begin
    store=Sen.create_vector_store(4;initial_capacity=16,)

    for index in 1:16
        Sen.insert_vector!(store,fill(Float32(index),4))
    end

    mktempdir() do path
        Sen.save_vector_store(path,store)
        result=run_vector_load_benchmark(path;repetitions=2,)

        @test result.file_bytes>0
        @test result.dim==4
        @test result.count==16
        @test !result.copied.mapped
        @test result.mapped.mapped
        @test result.copied.warm.p50_ms>=0
        @test result.mapped.warm.p50_ms>=0
        @test result.warm_speedup>0
        @test result.allocation_reduction>0
        @test_throws ArgumentError run_vector_load_benchmark(path;repetitions=0,)
    end
end

@testset "synthetic queries" begin
    queries=generate_synthetic_queries(8,4;seed=43,)

    @test size(queries)==(4,8)
    @test queries==generate_synthetic_queries(8,4;seed=43,)
    @test all(index->isapprox(sum(abs2,queries[:,index]),1.0f0;atol=1.0f-5,),1:8)
    @test queries!=generate_synthetic_queries(8,4;seed=44,)

    @test_throws ArgumentError generate_synthetic_queries(0,4)
    @test_throws ArgumentError generate_synthetic_queries(8,0)
end

@testset "clustered dataset" begin
    dataset=generate_clustered_dataset(100,8;clusters=4,cluster_noise=0.05,cluster_skew=1.0,seed=42,)
    query_set=generate_clustered_queries(10,dataset.centers;query_noise=0.05,cluster_skew=1.0,seed=43,)

    @test size(dataset.vectors)==(8,100)
    @test size(dataset.centers)==(8,4)
    @test length(dataset.assignments)==100
    @test Set(dataset.assignments)==Set(1:4)
    @test all(index->isapprox(sum(abs2,dataset.vectors[:,index]),1.0f0;atol=1.0f-5,),1:100)
    @test size(query_set.queries)==(8,10)
    @test all(cluster->1<=cluster<=4,query_set.assignments)
    @test dataset.vectors==generate_clustered_dataset(100,8;clusters=4,cluster_noise=0.05,cluster_skew=1.0,seed=42,).vectors

    @test_throws ArgumentError generate_clustered_dataset(10,4;clusters=11,)
    @test_throws ArgumentError generate_clustered_queries(0,dataset.centers)
end

@testset "fvecs loader" begin
    mktempdir() do path
        fvecs_path=joinpath(path,"tiny.fvecs")

        open(fvecs_path,"w") do io
            for vector in (Float32[3,4],Float32[0,2],Float32[-1,0])
                write(io,Int32(length(vector)))
                write(io,vector)
            end
        end

        vectors=load_fvecs(fvecs_path;limit=2,)

        @test size(vectors)==(2,2)
        @test vectors[:,1]≈Float32[0.6,0.8]
        @test vectors[:,2]≈Float32[0.0,1.0]
        @test load_fvecs(fvecs_path;normalize=false,)==Float32[3 0 -1;4 2 0]
        @test_throws ArgumentError load_fvecs(fvecs_path;limit=0,)
    end
end

@testset "recall at k" begin
    truth = [1,2,3,4]

    @test recall_at_k([1,2,3,4],truth,4)==1.0
    @test recall_at_k([1,3,8,9],truth,4)==0.5
    @test recall_at_k([8,9,10,11],truth,4)==0.0

    @test recall_at_k([1,2],truth,2)==1.0
    @test recall_at_k(Int[],Int[],4)==1.0
    @test recall_at_k([1],Int[],4)==0.0
    @test_throws ArgumentError recall_at_k([1],[1],0)
    end

@testset "exact ground truth" begin
    vectors = Float32[
        1 0 -1
        0 1 0
    ]
    metadata=[
        (name="right",),
        (name="up",),
        (name="left",),
    ]
    queries = Float32[
        1 0
        0 1
    ]
    truth = compute_groundtruth(vectors,metadata,queries;k=1,)
    @test length(truth)==2
    @test truth[1]==[1]
    @test truth[2]==[2]

    filtered_truth = compute_groundtruth(vectors,metadata,queries[:,1:1];k=1,filters=[(name="up",)],)
    @test filtered_truth[1]==[2]

    end
@testset "latency metrics" begin
    times_ns=[1_000_000,2_000_000,3_000_000,4_000_000,5_000_000]
    summary=latency_summary(times_ns)

    @test summary.minimum_ms==1.0
    @test summary.mean_ms==3.0
    @test summary.p50_ms==3.0
    @test summary.p95_ms==5.0
    @test summary.maximum_ms==5.0

    measurement=measure_latency(()->sum(1:100);repetitions=3,)

    @test measurement.result==5050
    @test length(measurement.times_ns)==3
    @test all(time->time>=0,measurement.times_ns)

    @test_throws ArgumentError latency_summary(Int[])
    @test_throws ArgumentError measure_latency(()->nothing;repetitions=0,)
end

@testset "benchmark runner" begin
    vectors=Float32[
        -1.0 -0.9 0.9 1.0
        0.0 0.1 0.1 0.0
    ]

    metadata=[
        (side="left",),
        (side="left",),
        (side="right",),
        (side="right",),
    ]

    queries=reshape(Float32[-1.0,0.0],2,1)

    filters=[(side="right",)]

    results=run_benchmark(vectors,metadata,queries,filters;nlists=2,k=2,nprobe=1,iterations=20,seed=42,repetitions=2,vector_weight=0.0,filter_weight=1.0,)

    @test results.exact.average_recall==1.0
    @test results.ivf_prefilter.average_recall==0.0
    @test results.ivf_postfilter.average_recall==0.0
    @test results.filter_aware.average_recall==1.0
    @test results.filter_aware_bound.average_recall==1.0

    @test results.exact.average_results==2.0
    @test results.ivf_prefilter.average_results==0.0
    @test results.ivf_postfilter.average_results==0.0
    @test results.filter_aware.average_results==2.0

    @test results.exact.average_candidates_scored==2.0
    @test results.ivf_prefilter.average_candidates_visited==0.0
    @test results.ivf_prefilter.average_candidates_scored==0.0
    @test results.ivf_postfilter.average_candidates_scored==2.0
    @test results.filter_aware.average_candidates_scored==2.0
    @test results.filter_aware.average_candidates_visited==2.0
    @test results.filter_aware_bound.average_candidates_scored==2.0

    @test results.average_selectivity==0.5
    @test results.exact.latency.p50_ms>=0
    @test results.filter_aware.latency.p50_ms>=0
    @test results.filter_aware_bound.latency.p50_ms>=0

    dot_results=run_benchmark(vectors,metadata,queries,filters;nlists=2,k=2,nprobe=1,iterations=5,seed=42,repetitions=1,metric=:dot,vector_weight=0.0,filter_weight=1.0,)
    @test dot_results.filter_aware_bound===nothing
end

@testset "candidate counts" begin
    vectors=Float32[
        -1.0 -0.9 0.9 1.0
        0.0 0.1 0.1 0.0
    ]

    metadata=[
        (side="left",),
        (side="left",),
        (side="right",),
        (side="right",),
    ]

    query=Float32[-1.0,0.0]

    index=build_filter_aware_ivf(vectors,metadata;nlists=2,iterations=20,seed=42,)
    filter_index=build_bitset_index(metadata)

    @test count_exact_candidates(filter_index,nothing)==4
    @test count_exact_candidates(filter_index,(side="right",))==2
    @test count_exact_candidates(filter_index,(side="missing",))==0

    @test count_ivf_candidates(index.ivf,query;nprobe=1,)==2
    @test count_ivf_candidates(index.ivf,query;nprobe=2,)==4

    prefilter_work=count_ivf_prefilter_work(index.ivf,metadata,query,(side="right",);nprobe=1,filter_index=filter_index,)

    @test prefilter_work.visited==2
    @test prefilter_work.scored==0
    @test prefilter_work.probed_lists==1

    @test count_filter_aware_candidates(index,metadata,query,(side="right",);nprobe=1,vector_weight=0.0,filter_weight=1.0,)==2
    @test count_filter_aware_candidates(index,metadata,query,(side="missing",);nprobe=1,)==0

    @test_throws ArgumentError count_ivf_candidates(index.ivf,query;nprobe=0,)
    @test_throws DimensionMismatch count_ivf_candidates(index.ivf,Float32[1.0];nprobe=1,)
end

@testset "selectivity metadata" begin
    metadata=generate_selectivity_metadata(100,0.10;seed=42,)

    @test length(metadata)==100
    @test count(row->row.selected,metadata)==10
    @test metadata==generate_selectivity_metadata(100,0.10;seed=42,)

    order=generate_selectivity_order(100;seed=42,)
    sparse=generate_selectivity_metadata(order,0.10)
    dense=generate_selectivity_metadata(order,0.50)

    @test all(!sparse[index].selected||dense[index].selected for index in 1:100)
    @test count(row->row.selected,dense)==50

    @test_throws ArgumentError generate_selectivity_metadata(0,0.10,)
    @test_throws ArgumentError generate_selectivity_metadata(100,-0.10,)
    @test_throws ArgumentError generate_selectivity_metadata(100,1.10,)
    @test_throws ArgumentError generate_selectivity_metadata([1,1],0.50)
end

@testset "filter workloads" begin
    dataset=generate_synthetic_dataset(100,8;seed=42,)

    for workload in (:random,:correlated,:anticorrelated,:skewed)
        sparse=generate_filter_metadata(dataset.vectors,0.10;workload=workload,seed=43,)
        dense=generate_filter_metadata(dataset.vectors,0.50;workload=workload,seed=43,)

        @test count(row->row.selected,sparse)==10
        @test count(row->row.selected,dense)==50
        @test all(!sparse[index].selected||dense[index].selected for index in 1:100)
    end

    @test_throws ArgumentError generate_filter_order(dataset.vectors;workload=:missing,)
    @test_throws ArgumentError generate_filter_order(dataset.vectors;skew=0.0,)
end

@testset "selectivity sweep" begin
    dataset=generate_synthetic_dataset(40,4;seed=42,)
    queries=generate_synthetic_queries(2,4;seed=43,)

    results=run_selectivity_sweep(
        dataset.vectors,
        queries,
        [0.10,0.50];
        nlists=4,
        k=3,
        nprobe=2,
        iterations=5,
        seed=42,
        repetitions=1,
    )

    @test length(results)==2
    @test results[1].target_selectivity==0.10
    @test results[2].target_selectivity==0.50
    @test results[1].actual_selectivity==0.10
    @test results[2].actual_selectivity==0.50

    for result in results
        @test result.benchmark.exact.average_recall==1.0
        @test result.benchmark.exact.average_candidates_scored>=0
        @test result.benchmark.ivf_prefilter.average_candidates_visited>=result.benchmark.ivf_prefilter.average_candidates_scored
        @test result.benchmark.ivf_postfilter.average_candidates_scored>=0
        @test result.benchmark.filter_aware.average_candidates_scored>=0
        @test result.benchmark.exact.latency.p50_ms>=0
        @test result.benchmark.filter_aware.latency.p95_ms>=0
    end

    @test_throws ArgumentError run_selectivity_sweep(dataset.vectors,queries,Float64[];nlists=4,)
end

@testset "benchmark empty matches" begin
    vectors=Float32[
        -1.0 -0.9 0.9 1.0
        0.0 0.1 0.1 0.0
    ]

    metadata=fill((selected=false,),4)
    queries=reshape(Float32[-1.0,0.0],2,1)
    filters=[(selected=true,)]

    results=run_benchmark(
        vectors,
        metadata,
        queries,
        filters;
        nlists=2,
        k=2,
        nprobe=1,
        iterations=5,
        seed=42,
        repetitions=1,
    )

    @test results.exact.average_recall==1.0
    @test results.ivf_prefilter.average_recall==1.0
    @test results.ivf_postfilter.average_recall==1.0
    @test results.filter_aware.average_recall==1.0

    @test results.exact.average_results==0.0
    @test results.ivf_prefilter.average_results==0.0
    @test results.ivf_postfilter.average_results==0.0
    @test results.filter_aware.average_results==0.0
end

@testset "nprobe sweep" begin
    dataset=generate_synthetic_dataset(40,4;seed=42,)
    queries=generate_synthetic_queries(2,4;seed=43,)
    metadata=generate_selectivity_metadata(40,0.50;seed=44,)
    filters=fill((selected=true,),2)

    results=run_nprobe_sweep(
        dataset.vectors,
        metadata,
        queries,
        filters,
        [1,2,4];
        nlists=4,
        k=3,
        iterations=5,
        repetitions=1,
    )

    @test [result.nprobe for result in results]==[1,2,4]
    @test length(sweep_points(results,:ivf_prefilter))==3
    @test all(point->point.candidates_visited>=point.candidates_scored,sweep_points(results,:ivf_prefilter))
    @test all(point->point.recall>=0,pareto_frontier(results,:filter_aware))
    @test best_at_recall(results,:exact,1.0)!==nothing
    @test_throws ArgumentError best_at_recall(results,:ivf_prefilter,1.1)

    @test_throws ArgumentError run_nprobe_sweep(dataset.vectors,metadata,queries,filters,Int[];nlists=4,)
end
