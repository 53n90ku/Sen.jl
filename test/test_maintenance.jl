using Test
using Sen

@testset "maintenance config" begin
    config=MaintenanceConfig()

    @test config.enabled
    @test config.minimum_changes==1_000
    @test config.delta_threshold==10_000
    @test config.delta_ratio==0.10
    @test config.max_delta_search_records==20_000
    @test config.active_segment_threshold==10_000
    @test config.incremental_indexing
    @test config.tombstone_threshold==10_000
    @test config.tombstone_ratio==0.10
    @test config.max_retries==3
    @test config.retry_delay_ms==50
    @test config.persist_after_rebuild
    @test MaintenanceConfig(delta_ratio=0,tombstone_ratio=0,).delta_ratio==0.0
    @test_throws ArgumentError MaintenanceConfig(minimum_changes=0,)
    @test_throws ArgumentError MaintenanceConfig(delta_threshold=-1,)
    @test_throws ArgumentError MaintenanceConfig(delta_ratio=1.1,)
    @test_throws ArgumentError MaintenanceConfig(max_delta_search_records=0,)
    @test_throws ArgumentError MaintenanceConfig(active_segment_threshold=0,)
    @test_throws ArgumentError MaintenanceConfig(max_delta_search_records=2,active_segment_threshold=3,)
    @test_throws ArgumentError MaintenanceConfig(tombstone_threshold=-1,)
    @test_throws ArgumentError MaintenanceConfig(tombstone_ratio=-0.1,)
    @test_throws ArgumentError MaintenanceConfig(max_retries=-1,)
    @test_throws ArgumentError MaintenanceConfig(retry_delay_ms=-1,)
end

@testset "index generation rebases concurrent mutations" begin
    config=MaintenanceConfig(enabled=false,max_delta_search_records=64,persist_after_rebuild=false,)
    db=create_db("generation-rebase";dim=2,durable=false,maintenance_config=config,)
    insert!(db,Float32[1 0 -1;0 1 0],[(name="right",),(name="up",),(name="left",)];ids=["right","up","left"],)
    build!(db;nlists=1,iterations=3,seed=21,)
    snapshot_ready=Channel{UInt64}(1)
    release_build=Channel{Nothing}(1)
    hook=snapshot->begin
        put!(snapshot_ready,snapshot.revision)
        take!(release_build)
    end
    builder=Threads.@spawn build!(db;nlists=1,iterations=4,seed=22,_snapshot_hook=hook,)
    build_revision=take!(snapshot_ready)

    insert!(db,[0.7,0.7],(name="diagonal",);id="diagonal",)
    update!(db,"right";vector=[0.9,0.1],metadata=(name="right-new",),)
    delete!(db,"up")
    put!(release_build,nothing)
    fetch(builder)

    @test db.index_revision==build_revision
    @test db.revision==build_revision+3
    @test !is_built(db)
    @test Sen.delta_search_work(db)==2
    @test [entry.revision for entry in db.mutation_history]==collect(build_revision+1:db.revision)
    @test get_record(db,"right").metadata==(name="right-new",)
    @test get_record(db,"diagonal").metadata==(name="diagonal",)
    @test_throws KeyError get_record(db,"up")
    @test Set(result.id for result in search(db,[1.0,0.0];k=3,strategy=:ivf,nprobe=1,))==Set(["right","left","diagonal"])

    rebuild!(db)
    @test is_built(db)
    @test isempty(db.mutation_history)
    close(db)
end

@testset "delta search work is hard bounded" begin
    config=MaintenanceConfig(enabled=false,max_delta_search_records=3,persist_after_rebuild=false,)
    db=create_db("bounded-delta";dim=2,durable=false,maintenance_config=config,)
    initial_vectors=Matrix{Float32}(undef,2,12)

    for id in 1:12
        initial_vectors[:,id].=Float32[1,id/20]
    end

    insert!(db,initial_vectors,[(number=id,) for id in 1:12];ids=collect(1:12),)
    build!(db;nlists=2,iterations=3,seed=31,)

    for id in 13:30
        insert!(db,Float32[1,id/20],(number=id,);id=id,)
        @test Sen.delta_search_work(db)<=config.max_delta_search_records
        @test !isempty(search(db,Float32[1,0];k=3,strategy=:ivf,nprobe=2,))
    end

    segment_status=Sen.wait_for_segment_indexing(db;timeout=10.0,)
    @test segment_status.status==:completed
    @test Sen.delta_search_work(db)==0
    @test length(db.immutable_segments)==7
    @test all(segment->segment.index!==nothing,db.immutable_segments)
    revision=db.revision
    oversized=Float32[1 1 1 1;2 3 4 5]
    @test_throws ArgumentError insert!(db,oversized,[(number=id,) for id in 31:34];ids=collect(31:34),)
    @test db.revision==revision
    @test length(db)==30
    @test Sen.delta_search_work(db)==0

    queries=Float32[1 1;0 1]
    @test length(search(db,queries;k=3,parallel=false,strategy=:exact,))==2
    info=database_info(db)
    @test info.delta_search_work==Sen.delta_search_work(db)
    @test info.delta_search_limit==3

    configure_maintenance!(db,MaintenanceConfig(enabled=false,max_delta_search_records=1,persist_after_rebuild=false,))
    @test Sen.delta_search_work(db)==0
    @test database_info(db).delta_search_limit==1
    close(db)
end

@testset "automatic delta maintenance" begin
    config=MaintenanceConfig(minimum_changes=1,delta_threshold=1,delta_ratio=0.0,tombstone_threshold=0,tombstone_ratio=0.0,retry_delay_ms=1,persist_after_rebuild=false,)
    db=create_db("maintenance-delta";dim=2,durable=false,maintenance_config=config,)

    insert!(db,Float32[1 0 1 0;0 1 1 -1],[(name="right",),(name="up",),(name="diagonal",),(name="down",)];ids=[1,2,3,4],)
    build!(db;nlists=2,iterations=3,seed=11,)
    insert!(db,Float32[-1,0],(name="left",);id=5,)

    status=wait_for_maintenance(db;timeout=10.0,)

    @test status.status==:completed
    @test status.delta_count==0
    @test status.tombstone_count==0
    @test status.last_completed_revision==db.revision
    @test is_built(db)
    @test get_record(db,5).metadata==(name="left",)
    close(db)
end

@testset "automatic tombstone maintenance" begin
    config=MaintenanceConfig(minimum_changes=1,delta_threshold=0,delta_ratio=0.0,tombstone_threshold=1,tombstone_ratio=0.0,retry_delay_ms=1,persist_after_rebuild=false,)
    db=create_db("maintenance-tombstone";dim=2,durable=false,maintenance_config=config,)

    insert!(db,Float32[1 0 -1;0 1 0],[(name="right",),(name="up",),(name="left",)];ids=[1,2,3],)
    build!(db;nlists=3,iterations=3,seed=7,)
    delete!(db,1)

    status=wait_for_maintenance(db;timeout=10.0,)

    @test status.status==:completed
    @test status.base_count==2
    @test status.tombstone_count==0
    @test is_built(db)
    @test_throws KeyError get_record(db,1)
    @test get_record(db,2).metadata==(name="up",)
    close(db)
end

@testset "maintenance preserves concurrent writes" begin
    config=MaintenanceConfig(minimum_changes=1,delta_threshold=1,delta_ratio=0.0,tombstone_threshold=1,tombstone_ratio=0.0,max_retries=20,retry_delay_ms=1,persist_after_rebuild=false,)
    db=create_db("maintenance-concurrent";dim=8,durable=false,maintenance_config=config,)
    vectors=rand(Float32,8,2_000)
    metadata=[(number=index,) for index in 1:2_000]

    insert!(db,vectors,metadata;ids=collect(1:2_000),)
    build!(db;nlists=16,iterations=8,seed=9,)
    insert!(db,rand(Float32,8),(number=2_001,);id=2_001,)

    for id in 2_002:2_025
        insert!(db,rand(Float32,8),(number=id,);id=id,)
        yield()
    end

    status=wait_for_maintenance(db;timeout=30.0,)

    @test status.status==:completed
    @test status.attempts>=1
    @test status.delta_count==0
    @test status.tombstone_count==0
    @test length(db)==2_025
    @test is_built(db)

    for id in 2_001:2_025
        @test get_record(db,id).metadata==(number=id,)
    end

    close(db)
end

@testset "durable maintenance snapshot" begin
    mktempdir() do path
        config=MaintenanceConfig(minimum_changes=1,delta_threshold=1,delta_ratio=0.0,tombstone_threshold=0,tombstone_ratio=0.0,retry_delay_ms=1,persist_after_rebuild=true,)
        db=create_db(path;dim=2,maintenance_config=config,checkpoint_retain_snapshots=1,)

        insert!(db,Float32[1 0;0 1],[(name="right",),(name="up",)];ids=[1,2],)
        build!(db;nlists=1,iterations=3,seed=4,)
        save!(db;retain_snapshots=1,)
        insert!(db,Float32[-1,0],(name="left",);id=3,)
        status=wait_for_maintenance(db;timeout=10.0,)

        @test status.status==:completed
        @test is_built(db)
        @test length(database_snapshot_generations(path))==1
        close(db)

        loaded=load_db(path;maintenance_config=MaintenanceConfig(enabled=false,),)
        @test is_built(loaded)
        @test length(loaded)==3
        @test get_record(loaded,3).metadata==(name="left",)
        close(loaded)
    end
end
