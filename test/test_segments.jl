using Test
using Random
using Sen

function materialized_exact_results(db,query;k::Int,filter=nothing,)
    stores=Sen.materialize_database(db)
    converted=Sen.convert_validated_vector(query,db.dim,db.metric;context="query",)
    normalized_filter=Sen.normalize_filter(filter)
    raw=Sen.search_exact(
        Sen.stored_vectors(stores.vector_store),
        Sen.stored_metadata(stores.metadata_store),
        converted;
        k=k,
        metric=db.metric,
        filter=normalized_filter,
    )
    return Sen.database_search_results(raw,stores.id_store)
end

@testset "automatic sealing and background segment indexing" begin
    config=MaintenanceConfig(enabled=false,max_delta_search_records=4,active_segment_threshold=2,incremental_indexing=true,max_retries=0,persist_after_rebuild=false,)
    db=create_db("incremental-segment-index";dim=2,metric=:dot,durable=false,maintenance_config=config,)
    insert!(db,Float32[1 0.8;0 0],[(version=1,),(version=1,)];ids=[1,2],)
    build!(db;nlists=1,iterations=2,seed=601,)
    started=Channel{String}(1)
    release=Channel{Nothing}(1)
    released=false
    Sen.set_segment_index_build_hook!(db) do snapshot
        put!(started,snapshot.id)
        take!(release)
    end

    try
        insert!(db,Float32[0.7,0],(version=1,);id=3,)
        insert!(db,Float32[0.6,0],(version=1,);id=4,)
        @test timedwait(()->isready(started),10.0)==:ok
        segment_id=take!(started)

        @test length(db.immutable_segments)==2
        @test db.immutable_segments[2].id==segment_id
        @test db.immutable_segments[2].index===nothing
        @test Sen.active_segment_is_empty(db.active_segment)
        @test [result.id for result in search(db,Float32[1,0];k=4,nprobe=1,strategy=:ivf,)]==[1,2,3,4]

        insert!(db,Float32[0.5,0],(version=1,);id=5,)
        @test length(db.active_segment.store)==1
        @test get_record(db,5).metadata.version==1
        put!(release,nothing)
        released=true
        Sen.set_segment_index_build_hook!(db,nothing)
        status=Sen.wait_for_segment_indexing(db;timeout=10.0,)

        @test status.status==:completed
        @test status.completed_count==1
        @test db.immutable_segments[2].index!==nothing
        @test db.immutable_segments[2].build_config!==nothing
        @test [result.id for result in search(db,Float32[1,0];k=5,nprobe=1,strategy=:ivf,)]==[1,2,3,4,5]
    finally
        released||put!(release,nothing)
        Sen.set_segment_index_build_hook!(db,nothing)
        close(db)
    end
end

@testset "oversized active mutation is atomic" begin
    mktempdir() do path
        config=MaintenanceConfig(enabled=false,max_delta_search_records=2,active_segment_threshold=2,incremental_indexing=true,persist_after_rebuild=false,)
        db=create_db(path;dim=2,metric=:dot,maintenance_config=config,checkpoint_operations=0,checkpoint_bytes=0,)
        insert!(db,Float32[1 0.5;0 0],[(version=1,),(version=1,)];ids=[1,2],)
        build!(db;nlists=1,iterations=2,seed=602,)
        revision=db.revision
        wal_bytes=filesize(Sen.database_wal_path(path))

        @test_throws ArgumentError insert!(db,Float32[0.4 0.3 0.2;0 0 0],[(version=1,),(version=1,),(version=1,)];ids=[3,4,5],)
        @test db.revision==revision
        @test length(db)==2
        @test length(db.immutable_segments)==1
        @test Sen.active_segment_is_empty(db.active_segment)
        @test filesize(Sen.database_wal_path(path))==wal_bytes
        close(db)
    end
end

@testset "segment indexing failure keeps exact fallback" begin
    config=MaintenanceConfig(enabled=false,max_delta_search_records=1,active_segment_threshold=1,incremental_indexing=true,max_retries=0,persist_after_rebuild=false,)
    db=create_db("segment-index-failure";dim=2,metric=:dot,durable=false,maintenance_config=config,)
    insert!(db,Float32[1,0],(version=1,);id=1,)
    build!(db;nlists=1,iterations=2,seed=603,)
    Sen.set_segment_index_build_hook!(db,_ -> error("injected segment index failure"))
    insert!(db,Float32[0.8,0],(version=1,);id=2,)
    failed=Sen.wait_for_segment_indexing(db;timeout=10.0,)

    @test failed.status==:failed
    @test occursin("injected segment index failure",failed.last_error)
    @test db.immutable_segments[2].index===nothing
    @test [result.id for result in search(db,Float32[1,0];k=2,nprobe=1,strategy=:ivf,)]==[1,2]

    Sen.set_segment_index_build_hook!(db,nothing)
    Sen.maybe_schedule_segment_indexing!(db)
    recovered=Sen.wait_for_segment_indexing(db;timeout=10.0,)
    @test recovered.status==:completed
    @test db.immutable_segments[2].index!==nothing
    close(db)
end

@testset "segment indexing resumes and persists" begin
    mktempdir() do path
        paused=MaintenanceConfig(enabled=false,max_delta_search_records=2,active_segment_threshold=2,incremental_indexing=false,persist_after_rebuild=false,)
        db=create_db(path;dim=2,metric=:dot,maintenance_config=paused,)
        insert!(db,Float32[1 0.8;0 0],[(version=1,),(version=1,)];ids=[1,2],)
        build!(db;nlists=1,iterations=2,seed=604,)
        insert!(db,Float32[0.7,0],(version=1,);id=3,)
        insert!(db,Float32[0.6,0],(version=1,);id=4,)
        @test db.immutable_segments[2].index===nothing
        save!(db)
        close(db)

        running=MaintenanceConfig(enabled=false,max_delta_search_records=2,active_segment_threshold=2,incremental_indexing=true,persist_after_rebuild=false,)
        loaded=load_db(path;mmap_vectors=false,maintenance_config=running,)
        status=Sen.wait_for_segment_indexing(loaded;timeout=10.0,)
        @test status.status==:completed
        @test loaded.immutable_segments[2].index!==nothing
        save!(loaded)
        close(loaded)

        reopened=load_db(path;mmap_vectors=false,maintenance_config=paused,)
        @test reopened.immutable_segments[2].index!==nothing
        @test [result.id for result in search(reopened,Float32[1,0];k=4,nprobe=1,strategy=:ivf,)]==[1,2,3,4]
        close(reopened)
    end
end

@testset "immutable segment lifecycle" begin
    db=create_db("segment-lifecycle";dim=3,metric=:dot,durable=false,maintenance_config=MaintenanceConfig(enabled=false,incremental_indexing=false,),)
    insert!(db,Float32[1 0.8 0.6 0.4;0 0 0 0;0 0 0 0],[(group="red",version=1,),(group="blue",version=1,),(group="red",version=1,),(group="blue",version=1,)];ids=["a","b","c","d"],)
    build!(db;nlists=2,iterations=3,seed=501,)
    primary=first(db.immutable_segments)
    primary_vectors=copy(Sen.stored_vectors(primary.vector_store))
    primary_excluded=copy(primary.excluded)

    upsert!(db,Float32[0.95,0,0],(group="blue",version=2,);id="a",)
    insert!(db,Float32[0.9,0,0],(group="red",version=1,);id="e",)
    delete!(db,"b")

    @test Sen.stored_vectors(primary.vector_store)==primary_vectors
    @test primary.excluded==primary_excluded
    @test [result.id for result in search(db,Float32[1,0,0];k=10,strategy=:exact,)]==["a","e","c","d"]
    @test [result.id for result in search(db,Float32[1,0,0];k=10,filter=Sen.Eq(:group,"red"),strategy=:exact,)]==["e","c"]

    first_sealed=Sen.seal_active_segment!(db)
    @test first_sealed===db.immutable_segments[2]
    @test first_sealed.revision_start==UInt64(2)
    @test first_sealed.revision_end==UInt64(4)
    @test first_sealed.filter_index!==nothing
    @test first_sealed.index===nothing
    @test Sen.active_segment_is_empty(db.active_segment)

    first_sealed_vectors=copy(Sen.stored_vectors(first_sealed.vector_store))
    first_sealed_tombstones=copy(first_sealed.tombstone_ids)
    update!(db,"e";vector=Float32[0.85,0,0],metadata=(group="blue",version=2,),)
    delete!(db,"a")
    insert!(db,Float32[0.75,0,0],(group="red",version=1,);id="f",)
    second_sealed=Sen.seal_active_segment!(db)

    @test Sen.stored_vectors(first_sealed.vector_store)==first_sealed_vectors
    @test first_sealed.tombstone_ids==first_sealed_tombstones
    @test second_sealed.revision_start==UInt64(5)
    @test second_sealed.revision_end==UInt64(7)
    @test length(db.immutable_segments)==3
    @test length(db)==4
    @test_throws KeyError get_record(db,"a")
    @test_throws KeyError get_record(db,"b")

    query=Float32[1,0,0]
    expected=materialized_exact_results(db,query;k=10,)
    actual=search(db,query;k=10,strategy=:exact,)
    @test [result.id for result in actual]==[result.id for result in expected]==["e","f","c","d"]
    @test [result.score for result in actual]==[result.score for result in expected]

    expected_red=materialized_exact_results(db,query;k=10,filter=Sen.Eq(:group,"red"),)
    actual_red=search(db,query;k=10,filter=Sen.Eq(:group,"red"),strategy=:exact,)
    @test [result.id for result in actual_red]==[result.id for result in expected_red]==["f","c"]

    queries=Float32[1 0;0 0;0 1]
    batch=search(db,queries;k=10,strategy=:exact,parallel=false,)
    @test [[result.id for result in results] for results in batch]==[[result.id for result in search(db,@view(queries[:,index]);k=10,strategy=:exact,)] for index in axes(queries,2)]
end

@testset "segment persistence and WAL replay" begin
    mktempdir() do path
        config=MaintenanceConfig(enabled=false,incremental_indexing=false,)
        db=create_db(path;dim=3,metric=:dot,checkpoint_operations=0,checkpoint_bytes=0,maintenance_config=config,)
        insert!(db,Float32[1 0.8 0.6;0 0 0;0 0 0],[(version=1,),(version=1,),(version=1,)];ids=["a","b","c"],)
        build!(db;nlists=2,iterations=3,seed=502,)
        insert!(db,Float32[0.7,0,0],(version=1,);id="d",)
        Sen.seal_active_segment!(db)
        save!(db)
        snapshot_path=Sen.current_database_snapshot(path)

        @test isfile(Sen.database_segment_manifest_path(snapshot_path))
        @test !isfile(joinpath(snapshot_path,"vectors.bin"))
        @test !isfile(joinpath(snapshot_path,"metadata.bin"))
        @test !isfile(joinpath(snapshot_path,"ids.bin"))

        upsert!(db,Float32[0.95,0,0],(version=2,);id="a",)
        delete!(db,"b")
        insert!(db,Float32[0.9,0,0],(version=1,);id="e",)
        expected_ids=[result.id for result in search(db,Float32[1,0,0];k=10,strategy=:exact,)]
        expected_revision=db.revision
        close(db)

        loaded=load_db(path;mmap_vectors=true,maintenance_config=config,)
        @test loaded.segment_mode
        @test length(loaded.immutable_segments)==2
        @test length(loaded.active_segment.store)==2
        @test length(loaded.active_segment.tombstone_ids)==1
        @test loaded.revision==expected_revision
        @test [result.id for result in search(loaded,Float32[1,0,0];k=10,strategy=:exact,)]==expected_ids
        @test get_record(loaded,"a").metadata.version==2
        @test_throws KeyError get_record(loaded,"b")
        @test all(segment->length(segment.vector_store)==0||Sen.is_mapped(segment.vector_store),loaded.immutable_segments)
        @test !Sen.is_mapped(loaded.active_segment.store.vector_store)

        save!(loaded)
        close(loaded)
        reopened=load_db(path;mmap_vectors=false,maintenance_config=config,)
        @test length(reopened.immutable_segments)==2
        @test reopened.revision==expected_revision
        @test [result.id for result in search(reopened,Float32[1,0,0];k=10,strategy=:exact,)]==expected_ids
        close(reopened)
    end
end

@testset "empty primary and tombstone-only segments" begin
    mktempdir() do path
        db=create_db(path;dim=2,metric=:dot,maintenance_config=MaintenanceConfig(enabled=false,incremental_indexing=false,),)
        Sen.enable_database_segment_mode!(db)
        insert!(db,Float32[1,0],(kind="temporary",);id="temporary",)
        Sen.seal_active_segment!(db)
        delete!(db,"temporary")
        tombstone_segment=Sen.seal_active_segment!(db)

        @test length(db.immutable_segments)==3
        @test isempty(tombstone_segment.id_store.ids)
        @test tombstone_segment.tombstone_ids==Set(Any["temporary"])
        @test isempty(search(db,Float32[1,0];k=10,strategy=:exact,))
        save!(db)
        close(db)

        loaded=load_db(path;mmap_vectors=false,maintenance_config=MaintenanceConfig(enabled=false,incremental_indexing=false,),)
        @test length(loaded)==0
        @test length(loaded.immutable_segments)==3
        @test loaded.immutable_segments[3].tombstone_ids==Set(Any["temporary"])
        @test_throws KeyError get_record(loaded,"temporary")
        close(loaded)
    end
end

@testset "randomized multi-segment exact oracle" begin
    for(metric,seed) in ((:cosine,503),(:dot,504))
        rng=MersenneTwister(seed)
        db=create_db("segment-oracle-$(metric)";dim=5,metric=metric,durable=false,maintenance_config=MaintenanceConfig(enabled=false,incremental_indexing=false,),)
        model=Dict{Int,NamedTuple}()

        for id in 1:12
            vector=randn(rng,Float32,5)
            metadata=(group=rand(rng,1:3),version=0,)
            insert!(db,vector,metadata;id=id,)
            model[id]=(vector=copy(vector),metadata=metadata,)
        end

        build!(db;nlists=3,iterations=3,seed=seed,)
        next_id=13

        for step in 1:60
            operation=isempty(model) ? :insert : rand(rng,(:insert,:update,:delete))

            if operation===:insert
                id=next_id
                next_id+=1
                vector=randn(rng,Float32,5)
                metadata=(group=rand(rng,1:3),version=step,)
                insert!(db,vector,metadata;id=id,)
                model[id]=(vector=copy(vector),metadata=metadata,)
            elseif operation===:update
                id=rand(rng,collect(keys(model)))
                vector=randn(rng,Float32,5)
                metadata=(group=rand(rng,1:3),version=step,)
                update!(db,id;vector=vector,metadata=metadata,)
                model[id]=(vector=copy(vector),metadata=metadata,)
            else
                id=rand(rng,collect(keys(model)))
                delete!(db,id)
                delete!(model,id)
            end

            if step%10==0
                Sen.seal_active_segment!(db)
                query=randn(rng,Float32,5)
                k=min(8,length(model)+1)
                expected=materialized_exact_results(db,query;k=k,)
                actual=search(db,query;k=k,strategy=:exact,)
                @test [result.id for result in actual]==[result.id for result in expected]
                @test [result.score for result in actual]≈[result.score for result in expected]

                group=rand(rng,1:3)
                expected_filtered=materialized_exact_results(db,query;k=k,filter=Sen.Eq(:group,group),)
                actual_filtered=search(db,query;k=k,filter=Sen.Eq(:group,group),strategy=:exact,)
                @test [result.id for result in actual_filtered]==[result.id for result in expected_filtered]
                @test Set(Sen.database_visible_ids(db))==Set(keys(model))

                for(id,value) in model
                    record=get_record(db,id)
                    @test record.metadata==value.metadata
                end
            end
        end
    end
end
