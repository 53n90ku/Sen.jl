using Test
using Sen

@testset "delta lifecycle" begin
    db=create_db("delta-db"; dim = 2, metric = :dot, durable = false)
    insert!(db, [1.0, 0.0], (group = "base", name = "near"); id = "near")
    insert!(db, [0.6, 0.0], (group = "base", name = "middle"); id = "middle")
    insert!(db, [0.2, 0.0], (group = "base", name = "low"); id = "low")
    insert!(db, [-0.3, 0.0], (group = "base", name = "far"); id = "far")
    build!(db; nlists = 2, iterations = 4, seed = 31)
    base_index=db.index
    base_filter_index=db.filter_index

    insert!(db, [0.8, 0.0], (group = "delta", name = "inserted"); id = "inserted")

    @test db.index===base_index
    @test db.filter_index===base_filter_index
    @test length(db.delta_store)==1
    @test !is_built(db)
    @test is_dirty(db)

    update!(
        db,
        "middle";
        vector = [0.9, 0.0],
        metadata = (group = "delta", name = "updated"),
    )
    results=search(db, [1.0, 0.0]; k = 4, strategy = :exact)
    ids=[result.id for result in results]

    @test db.index===base_index
    @test db.filter_index===base_filter_index
    @test ids==["near", "middle", "inserted", "low"]
    @test length(ids)==length(unique(ids))
    @test get_record(db, "middle").metadata.name=="updated"

    delete!(db, "middle")
    results=search(db, [1.0, 0.0]; k = 4, strategy = :exact)

    @test db.index===base_index
    @test db.filter_index===base_filter_index
    @test all(result->result.id!="middle", results)
    @test_throws KeyError get_record(db, "middle")
    @test length(results)==4

    delete!(db, "near")
    results=search(db, [1.0, 0.0]; k = 3, strategy = :exact)

    @test db.index===base_index
    @test length(results)==3
    @test [result.id for result in results]==["inserted", "low", "far"]

    rebuild!(db)

    @test is_built(db)
    @test !is_dirty(db)
    @test db.index_revision==db.revision
    @test length(db.delta_store)==0
    @test !any(db.base_tombstones)
    @test Set(db.id_store.ids)==Set(Any["inserted", "low", "far"])
    @test [result.id for result in search(db, [1.0, 0.0]; k = 3, strategy = :exact)]==["inserted", "low", "far"]
end

@testset "delta filters" begin
    db=create_db("delta-filter-db"; dim = 2, metric = :dot, durable = false)
    insert!(db, [1.0, 0.0], (group = "red", name = "red-base"); id = "red-base")
    insert!(db, [0.9, 0.0], (group = "blue", name = "blue-base"); id = "blue-base")
    insert!(db, [0.8, 0.0], (group = "red", name = "moving"); id = "moving")
    insert!(db, [0.7, 0.0], (group = "blue", name = "blue-low"); id = "blue-low")
    build!(db; nlists = 2, iterations = 4, seed = 32)

    insert!(db, [0.95, 0.0], (group = "red", name = "red-delta"); id = "red-delta")
    insert!(db, [0.85, 0.0], (group = "blue", name = "blue-delta"); id = "blue-delta")
    update!(db, "moving"; vector = [0.99, 0.0], metadata = (group = "blue", name = "moved"))

    red=search(db, [1.0, 0.0]; k = 5, filter = (group = "red",), strategy = :exact)
    blue=search(db, [1.0, 0.0]; k = 5, filter = (group = "blue",), strategy = :exact)

    @test [result.id for result in red]==["red-base", "red-delta"]
    @test [result.id for result in blue]==["moving", "blue-base", "blue-delta", "blue-low"]
    @test all(result->result.metadata.group=="red", red)
    @test all(result->result.metadata.group=="blue", blue)
end

@testset "dirty delta persistence" begin
    mktempdir() do path
        db=create_db(path; dim = 2, metric = :dot)
        insert!(db, [0.8, 0.0], (state = "base",); id = "keep")
        insert!(db, [0.7, 0.0], (state = "old",); id = "update")
        insert!(db, [1.0, 0.0], (state = "delete",); id = "delete")
        build!(db; nlists = 2, iterations = 4, seed = 33)
        update!(db, "update"; vector = [0.95, 0.0], metadata = (state = "updated",))
        delete!(db, "delete")
        insert!(db, [0.9, 0.0], (state = "inserted",); id = "insert")

        @test is_dirty(db)
        @test db.index!==nothing
        @test length(db.delta_store)==2
        @test count(db.base_tombstones)==2

        save!(db)
        close(db)
        loaded=load_db(path)
        results=search(loaded, [1.0, 0.0]; k = 5, strategy = :exact)

        @test length(loaded)==3
        @test is_dirty(loaded)
        @test loaded.index!==nothing
        @test loaded.index_revision<loaded.revision
        @test length(loaded.delta_store)==2
        @test count(loaded.base_tombstones)==2
        @test [result.id for result in results]==["update", "insert", "keep"]
        @test get_record(loaded, "update").metadata.state=="updated"
        @test get_record(loaded, "insert").metadata.state=="inserted"
        @test_throws KeyError get_record(loaded, "delete")
        close(loaded)
    end
end

@testset "delta batch changes" begin
    db=create_db("delta-batch-db"; dim = 2, metric = :dot, durable = false)
    vectors=Float32[
        1.0 0.8 0.6 0.4
        0.0 0.0 0.0 0.0
    ]
    metadata=[
        (name = "base-1", group = "base"),
        (name = "base-2", group = "base"),
        (name = "base-3", group = "base"),
        (name = "base-4", group = "base"),
    ]
    insert!(db, vectors, metadata; ids = ["base-1", "base-2", "base-3", "base-4"])
    build!(db; nlists = 2, iterations = 4, seed = 34)
    base_index=db.index

    inserted=insert!(
        db,
        Float32[0.9 0.7; 0.0 0.0],
        [(name = "delta-1", group = "delta"), (name = "delta-2", group = "delta")];
        ids = ["delta-1", "delta-2"],
    )

    @test inserted==["delta-1", "delta-2"]
    @test db.index===base_index
    @test length(db.delta_store)==2

    upserted=upsert!(
        db,
        Float32[0.95 0.85 0.75; 0.0 0.0 0.0],
        [
            (name = "base-1-new", group = "new"),
            (name = "delta-1-new", group = "new"),
            (name = "new", group = "new"),
        ];
        ids = ["base-1", "delta-1", "new"],
    )

    @test upserted==["base-1", "delta-1", "new"]
    @test db.index===base_index
    @test length(db.delta_store)==4
    @test count(db.base_tombstones)==1

    delete!(db, ["base-2", "delta-1"])
    results=search(db, [1.0, 0.0]; k = 10, strategy = :exact)
    ids=[result.id for result in results]

    @test db.index===base_index
    @test length(db)==5
    @test length(db.delta_store)==3
    @test count(db.base_tombstones)==2
    @test ids==["base-1", "new", "delta-2", "base-3", "base-4"]
    @test length(ids)==length(unique(ids))
    @test all(id->id!="base-2"&&id!="delta-1", ids)
    @test get_record(db, "base-1").metadata.name=="base-1-new"
end
