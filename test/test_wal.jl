using Test
using Sen

@testset "database WAL replay" begin
    mktempdir() do path
        db=create_db(path; dim = 2, metric = :dot)
        insert!(db, [1.0, 0.0], (name = "keep", version = 1); id = "keep")
        insert!(db, [0.8, 0.0], (name = "update", version = 1); id = "update")
        insert!(db, [0.6, 0.0], (name = "delete", version = 1); id = "delete")

        @test isfile(database_wal_path(path))
        @test length(read_database_wal(path).records)==3

        save!(db)
        snapshot_revision=db.revision
        checkpoint=read_database_wal(path)

        @test checkpoint.header.revision==snapshot_revision
        @test isempty(checkpoint.records)

        insert!(db, [0.9, 0.0], (name = "insert", version = 1); id = "insert")
        upsert!(
            db,
            Float32[0.95 0.7; 0.0 0.0],
            [(name = "update", version = 2), (name = "upsert", version = 1)];
            ids = ["update", "upsert"],
        )
        update!(db, "insert"; metadata = (name = "insert", version = 2))
        delete!(db, "delete")

        wal=read_database_wal(path)

        @test [record.revision for record in wal.records]==collect((snapshot_revision+1):db.revision)
        @test !wal.incomplete_tail

        expected_revision=db.revision
        close(db)
        loaded=load_db(path)

        @test loaded.revision==expected_revision
        @test length(loaded)==4
        @test get_record(loaded, "keep").metadata.version==1
        @test get_record(loaded, "update").metadata.version==2
        @test get_record(loaded, "insert").metadata.version==2
        @test get_record(loaded, "upsert").metadata.version==1
        @test_throws KeyError get_record(loaded, "delete")

        save!(loaded)
        checkpoint=read_database_wal(path)

        @test checkpoint.header.revision==loaded.revision
        @test isempty(checkpoint.records)
        close(loaded)
        reloaded=load_db(path)
        @test length(reloaded)==4
        close(reloaded)
    end
end

@testset "database WAL crash boundaries" begin
    mktempdir() do path
        db=create_db(path; dim = 2, metric = :dot)
        @test_throws ArgumentError create_db(path; dim = 2, metric = :dot)
        insert!(db, [1.0, 0.0], (name = "before-snapshot",); id = "before-snapshot")

        expected_revision=db.revision
        close(db)
        recovered=load_db(path)

        @test recovered.revision==expected_revision
        @test get_record(recovered, "before-snapshot").metadata.name=="before-snapshot"
        close(recovered)
    end

    mktempdir() do path
        db=create_db(path; dim = 2, metric = :dot)
        insert!(db, [1.0, 0.0], (name = "base",); id = "base")
        save!(db)
        revision=db.revision+UInt64(1)

        append_database_wal_put!(
            db,
            revision,
            [Float32[0.5, 0.0]],
            [(name = "synced",)],
            ["synced"],
        )

        @test_throws ArgumentError insert!(
            db,
            [0.25, 0.0],
            (name = "stale-memory",);
            id = "stale-memory",
        )

        close(db)
        recovered=load_db(path)

        @test recovered.revision==revision
        @test get_record(recovered, "synced").metadata.name=="synced"
        close(recovered)
    end

    mktempdir() do path
        db=create_db(path; dim = 2, metric = :dot)
        insert!(db, [1.0, 0.0], (name = "base",); id = "base")
        save!(db)
        insert!(db, [0.5, 0.0], (name = "complete",); id = "complete")
        wal_path=database_wal_path(path)
        complete_size=filesize(wal_path)

        open(wal_path, "a") do io
            write(io, UInt8[0x01, 0x02, 0x03])
        end

        @test filesize(wal_path)==complete_size+3

        close(db)
        recovered=load_db(path)

        @test get_record(recovered, "complete").metadata.name=="complete"
        @test filesize(wal_path)==complete_size
        @test !read_database_wal(path).incomplete_tail

        insert!(recovered, [0.25, 0.0], (name = "after-repair",); id = "after-repair")
        close(recovered)
        reloaded=load_db(path)

        @test get_record(reloaded, "after-repair").metadata.name=="after-repair"
        close(reloaded)
    end

    mktempdir() do path
        db=create_db(path; dim = 2, metric = :dot)
        insert!(db, [1.0, 0.0], (name = "base",); id = "base")
        save!(db)
        insert!(db, [0.5, 0.0], (name = "corrupt",); id = "corrupt")
        wal=read_database_wal(path)
        wal_path=database_wal_path(path)
        body_position=wal.header.header_bytes+sizeof(UInt64)

        open(wal_path, "r+") do io
            seek(io, body_position)
            byte=read(io, UInt8)
            seek(io, body_position)
            write(io, byte⊻UInt8(1))
        end

        close(db)
        @test_throws ArgumentError load_db(path)
    end

    mktempdir() do path
        db=create_db(path; dim = 2, metric = :dot)
        insert!(db, [1.0, 0.0], (name = "base",); id = "base")
        save!(db)
        wal_path=database_wal_path(path)

        open(wal_path, "r+") do io
            seek(io, 10)
            byte=read(io, UInt8)
            seek(io, 10)
            write(io, byte⊻UInt8(1))
        end

        close(db)
        @test_throws ArgumentError load_db(path)
    end
end

@testset "indexed database WAL replay" begin
    mktempdir() do path
        db=create_db(path; dim = 2, metric = :dot)
        insert!(db, [1.0, 0.0], (name = "first",); id = "first")
        insert!(db, [0.8, 0.0], (name = "second",); id = "second")
        insert!(db, [0.6, 0.0], (name = "third",); id = "third")
        insert!(db, [0.4, 0.0], (name = "fourth",); id = "fourth")
        build!(db; nlists = 2, iterations = 4, seed = 81)
        save!(db)
        index_revision=db.index_revision

        update!(db, "second"; vector = [0.95, 0.0], metadata = (name = "second-new",))
        delete!(db, "first")
        insert!(db, [0.9, 0.0], (name = "inserted",); id = "inserted")

        close(db)
        loaded=load_db(path)
        results=search(loaded, [1.0, 0.0]; k = 10, strategy = :exact)

        @test loaded.index!==nothing
        @test loaded.index_revision==index_revision
        @test loaded.index_revision<loaded.revision
        @test is_dirty(loaded)
        @test length(loaded.delta_store)==2
        @test count(loaded.base_tombstones)==2
        @test [result.id for result in results]==["second", "inserted", "third", "fourth"]
        @test get_record(loaded, "second").metadata.name=="second-new"
        @test_throws KeyError get_record(loaded, "first")
        close(loaded)
    end
end

@testset "database WAL rejected mutation" begin
    mktempdir() do path
        db=create_db(path; dim = 2)
        insert!(db, [1.0, 0.0], (name = "base",); id = "base")
        save!(db)
        revision=db.revision
        wal_size=filesize(database_wal_path(path))

        @test_throws ArgumentError insert!(
            db,
            [0.0, 1.0],
            (unsupported = Dict("value"=>1),);
            id = "invalid",
        )
        @test db.revision==revision
        @test length(db)==1
        @test filesize(database_wal_path(path))==wal_size
        @test_throws KeyError get_record(db, "invalid")
    end
end

@testset "database mutation fault atomicity" begin
    mktempdir() do path
        db=create_db(path; dim = 2, checkpoint_operations = 0, checkpoint_bytes = 0)
        insert!(db, [1.0, 0.0], (name = "base",); id = "base")
        build!(db; nlists = 1, iterations = 2)
        save!(db)
        revision=db.revision
        wal_size=filesize(database_wal_path(path))

        @test is_built(db)
        @test_throws ArgumentError insert!(
            db,
            [NaN, 1.0],
            (name = "invalid",);
            id = "invalid",
        )
        @test db.revision==revision
        @test filesize(database_wal_path(path))==wal_size

        Sen.inject_database_mutation_fault!(:before_wal)
        @test_throws Sen.InjectedDatabaseMutationError insert!(
            db,
            [0.0, 1.0],
            (name = "before",);
            id = "before",
        )
        @test db.revision==revision
        @test length(db)==1
        @test filesize(database_wal_path(path))==wal_size
        @test_throws KeyError get_record(db, "before")

        Sen.inject_database_mutation_fault!(:after_wal)
        after_wal_error=try
            insert!(db, [0.0, 1.0], (name = "after",); id = "after")
            nothing
        catch error
            error
        end

        @test after_wal_error isa Sen.DatabaseMutationCommittedError
        @test after_wal_error.cause isa Sen.InjectedDatabaseMutationError
        @test db.revision==revision+1
        @test get_record(db, "after").metadata.name=="after"
        @test db.index!==nothing
        @test !is_built(db)

        Sen.inject_database_mutation_fault!(:during_apply)
        during_apply_error=try
            insert!(
                db,
                Float32[-1.0 0.5; 0.0 0.5],
                [(name = "batch-a",), (name = "batch-b",)];
                ids = ["batch-a", "batch-b"],
            )
            nothing
        catch error
            error
        end

        @test during_apply_error isa Sen.DatabaseMutationCommittedError
        @test Set([get_record(db, "batch-a").id, get_record(db, "batch-b").id])==Set([
            "batch-a",
            "batch-b",
        ])
        @test length(db)==4

        Sen.inject_database_mutation_fault!(:after_apply)
        after_apply_error=try
            delete!(db, "base")
            nothing
        catch error
            error
        end

        @test after_apply_error isa Sen.DatabaseMutationCommittedError
        @test_throws KeyError get_record(db, "base")
        expected_revision=db.revision
        expected_ids=Set(["after", "batch-a", "batch-b"])
        close(db)

        reopened=load_db(path)
        @test reopened.revision==expected_revision
        @test Set(
            result.id for
            result in search(reopened, [1.0, 0.0]; k = length(reopened), strategy = :exact)
        )==expected_ids
        close(reopened)
    end

    db=create_db("memory-fault-db"; dim = 2, durable = false)
    insert!(db, [1.0, 0.0], (name = "base",); id = "base")
    state=(revision = db.revision, count = length(db), ids = copy(stored_ids(db.id_store)))
    Sen.inject_database_mutation_fault!(:during_apply)

    @test_throws Sen.InjectedDatabaseMutationError insert!(
        db,
        Float32[0.0 -1.0; 1.0 0.0],
        [(name = "a",), (name = "b",)];
        ids = ["a", "b"],
    )
    @test db.revision==state.revision
    @test length(db)==state.count
    @test stored_ids(db.id_store)==state.ids
    @test_throws KeyError get_record(db, "a")
    @test_throws KeyError get_record(db, "b")
end

@testset "automatic WAL checkpoints" begin
    mktempdir() do path
        db=create_db(path; dim = 2, checkpoint_operations = 2, checkpoint_bytes = 0)

        insert!(db, [1.0, 0.0], (name = "first",); id = "first")
        @test !isfile(database_current_path(path))

        insert!(db, [0.8, 0.0], (name = "second",); id = "second")
        wal=read_database_wal(path)

        @test isfile(database_current_path(path))
        @test wal.header.revision==db.revision
        @test isempty(wal.records)

        insert!(db, [0.6, 0.0], (name = "third",); id = "third")
        @test length(read_database_wal(path).records)==1
        close(db)
        loaded=load_db(path)
        @test length(loaded)==3
        close(loaded)
    end

    mktempdir() do path
        db=create_db(path; dim = 2, checkpoint_operations = 0, checkpoint_bytes = 1)
        insert!(db, [1.0, 0.0], (name = "bytes",); id = "bytes")

        @test isfile(database_current_path(path))
        @test isempty(read_database_wal(path).records)
    end

    mktempdir() do path
        db=create_db(path; dim = 2, checkpoint_operations = 0, checkpoint_bytes = 0)
        insert!(db, [1.0, 0.0], (name = "first",); id = "first")
        insert!(db, [0.8, 0.0], (name = "second",); id = "second")

        @test !isfile(database_current_path(path))
        @test length(read_database_wal(path).records)==2
    end

    mktempdir() do path
        db=create_db(
            path;
            dim = 2,
            checkpoint_operations = 1,
            checkpoint_bytes = 0,
            checkpoint_retain_snapshots = 1,
        )
        insert!(db, [1.0, 0.0], (name = "first",); id = "first")
        insert!(db, [0.8, 0.0], (name = "second",); id = "second")
        insert!(db, [0.6, 0.0], (name = "third",); id = "third")

        @test length(database_snapshot_generations(path))==1
        close(db)
        loaded=load_db(path)
        @test length(loaded)==3
        close(loaded)
    end

    mktempdir() do path
        @test_throws ArgumentError create_db(path; dim = 2, checkpoint_operations = -1)
        @test_throws ArgumentError create_db(path; dim = 2, checkpoint_bytes = -1)
        @test_throws ArgumentError create_db(path; dim = 2, checkpoint_retain_snapshots = 0)
    end
end
