using Test
using Sen

@testset "database info" begin
    config=MaintenanceConfig(enabled=false,)
    db=create_db("info-db";dim=2,metric=:dot,durable=false,maintenance_config=config,)
    empty_info=database_info(db)

    @test !ismutabletype(DatabaseInfo)
    @test empty_info isa DatabaseInfo
    @test empty_info.path=="info-db"
    @test empty_info.dim==2
    @test empty_info.metric==:dot
    @test empty_info.live_count==0
    @test empty_info.base_count==0
    @test empty_info.delta_count==0
    @test empty_info.delta_search_work==0
    @test empty_info.delta_search_limit==config.max_delta_search_records
    @test empty_info.tombstone_count==0
    @test empty_info.revision==0
    @test empty_info.index_revision===nothing
    @test empty_info.index_count==0
    @test empty_info.index_lists==0
    @test empty_info.index_bytes==0
    @test !empty_info.built
    @test !empty_info.dirty
    @test !empty_info.durable
    @test !empty_info.maintenance_enabled
    @test !empty_info.maintenance_running

    insert!(db,Float32[1 0 -1;0 1 0],[(name="right",),(name="up",),(name="left",)];ids=[1,2,3],)
    inserted_info=database_info(db)

    @test inserted_info.live_count==3
    @test inserted_info.base_count==3
    @test inserted_info.delta_count==0
    @test inserted_info.revision==1
    @test inserted_info.dirty
    @test !inserted_info.built

    build!(db;nlists=2,iterations=3,seed=2,)
    built_info=database_info(db)

    @test built_info.live_count==3
    @test built_info.base_count==3
    @test built_info.index_count==3
    @test built_info.index_lists==2
    @test built_info.index_bytes>0
    @test built_info.index_revision==built_info.revision
    @test built_info.built
    @test !built_info.dirty

    update!(db,1;vector=Float32[0.5,0.5],)
    delete!(db,2)
    dirty_info=database_info(db)

    @test dirty_info.live_count==2
    @test dirty_info.base_count==3
    @test dirty_info.delta_count==1
    @test dirty_info.delta_search_work==dirty_info.delta_count
    @test dirty_info.delta_search_work<=dirty_info.delta_search_limit
    @test dirty_info.tombstone_count==2
    @test dirty_info.delta_ratio==1/3
    @test dirty_info.tombstone_ratio==2/3
    @test dirty_info.index_count==3
    @test dirty_info.dirty
    @test !dirty_info.built
    @test dirty_info.live_count==dirty_info.base_count-dirty_info.tombstone_count+dirty_info.delta_count
    close(db)

    @test_throws ArgumentError database_info(db)
end

@testset "maintenance database info" begin
    config=MaintenanceConfig(minimum_changes=1,delta_threshold=1,delta_ratio=0,tombstone_threshold=0,tombstone_ratio=0,retry_delay_ms=1,persist_after_rebuild=false,)
    db=create_db("maintenance-info";dim=2,durable=false,maintenance_config=config,)

    insert!(db,Float32[1 0;0 1],[(name="right",),(name="up",)];ids=[1,2],)
    build!(db;nlists=1,iterations=3,seed=3,)
    insert!(db,Float32[-1,0],(name="left",);id=3,)
    wait_for_maintenance(db;timeout=10,)
    info=database_info(db)

    @test info.maintenance_enabled
    @test !info.maintenance_due
    @test info.maintenance_status==:completed
    @test !info.maintenance_running
    @test info.maintenance_attempts>=1
    @test info.last_rebuild_revision==info.revision
    @test info.last_rebuild_duration_ms>=0
    @test info.last_maintenance_error===nothing
    @test info.built
    close(db)
end

@testset "durable database info" begin
    mktempdir() do path
        db=create_db(path;dim=2,maintenance_config=MaintenanceConfig(enabled=false,),)
        insert!(db,Float32[1,0],(name="right",);id=1,)
        build!(db;nlists=1,iterations=3,seed=5,)
        save!(db)
        info=database_info(db)

        @test info.durable
        @test info.wal_revision==info.revision
        @test info.checkpoint_revision==info.revision
        @test info.index_bytes>0
        close(db)

        loaded=load_db(path;maintenance_config=MaintenanceConfig(enabled=false,),)
        loaded_info=database_info(loaded)

        @test loaded_info.durable
        @test loaded_info.built
        @test loaded_info.live_count==1
        @test loaded_info.index_count==1
        @test loaded_info.index_lists==1
        @test loaded_info.index_bytes>0
        @test loaded_info.index_revision==loaded_info.revision
        close(loaded)
    end
end
