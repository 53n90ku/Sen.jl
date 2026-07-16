using Test
using Random
using Sen

@testset "batch search workers" begin
    rng=MersenneTwister(91)
    db=create_db("batch-search";dim=8,durable=false,)
    vectors=randn(rng,Float32,8,160)
    metadata=[(group=index%4,name="record-$(index)",) for index in 1:160]
    ids=collect(1:160)

    insert!(db,vectors,metadata;ids=ids,)
    build!(db;nlists=8,iterations=4,seed=42,training_count=160,)

    queries=randn(rng,Float32,8,24)
    serial=search(db,queries;k=6,nprobe=4,strategy=:ivf,parallel=false,)
    parallel=search(db,queries;k=6,nprobe=4,strategy=:ivf,parallel=true,workers=3,parallel_threshold=2,)

    @test length(serial)==24
    @test [[result.id for result in results] for results in parallel]==[[result.id for result in results] for results in serial]
    @test [[result.score for result in results] for results in parallel]==[[result.score for result in results] for results in serial]
    @test Sen.resolve_database_batch_workers(24;parallel=false,workers=3,parallel_threshold=2,)==1
    @test Sen.resolve_database_batch_workers(1;parallel=true,workers=3,parallel_threshold=2,)==1
    @test Sen.resolve_database_batch_workers(24;parallel=true,workers=3,parallel_threshold=2,)==min(3,Threads.nthreads(:default))
    @test !Sen.database_batch_worker_owned()

    Sen.with_database_batch_worker() do
        @test Sen.database_batch_worker_owned()
        @test Sen.resolve_database_batch_workers(24;parallel=true,workers=3,parallel_threshold=2,)==1
    end

    @test !Sen.database_batch_worker_owned()

    empty_queries=zeros(Float32,8,0)
    @test isempty(search(db,empty_queries;parallel=true,workers=3,parallel_threshold=2,))
    @test_throws ArgumentError search(db,queries;parallel=true,workers=0,)
    @test_throws ArgumentError search(db,queries;parallel=true,parallel_threshold=0,)
    @test_throws ArgumentError search(db,queries;k=0,parallel=true,workers=3,parallel_threshold=2,)
    @test_throws ArgumentError search(db,queries;k=4,nprobe=4,max_nprobe=4,filter=(group=1,),strategy=:filter_aware,candidate_multiplier=-1.0,parallel=true,workers=3,parallel_threshold=2,)
    @test_throws DimensionMismatch search(db,zeros(Float32,7,4);parallel=true,workers=3,parallel_threshold=2,)

    insert!(db,randn(rng,Float32,8),(group=1,name="delta",);id="delta",)
    dirty_serial=search(db,queries;k=6,strategy=:exact,parallel=false,)
    dirty_parallel=search(db,queries;k=6,strategy=:exact,parallel=true,workers=3,parallel_threshold=2,)

    @test [[result.id for result in results] for results in dirty_parallel]==[[result.id for result in results] for results in dirty_serial]
end

@testset "parallel batch and writer" begin
    rng=MersenneTwister(92)
    db=create_db("batch-writer";dim=16,durable=false,)
    vectors=randn(rng,Float32,16,400)
    metadata=[(group=index%5,) for index in 1:400]

    insert!(db,vectors,metadata;ids=collect(1:400),)
    build!(db;nlists=16,iterations=4,seed=42,training_count=400,)

    queries=randn(rng,Float32,16,64)
    batch=Threads.@spawn search(db,queries;k=5,nprobe=8,strategy=:ivf,parallel=true,workers=4,parallel_threshold=2,)
    writer=Threads.@spawn update!(db,1;metadata=(group=1,),)
    results=fetch(batch)

    fetch(writer)

    @test length(results)==64
    @test all(result->length(result)==5,results)
    @test get_record(db,1).metadata.group==1
end
