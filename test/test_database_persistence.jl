using Test
using Sen

@testset "database persistence" begin
    mktempdir() do path
        db=create_db(path; dim = 2)

        insert!(db, [-1.0, 0.0], (side = "left", name = "left-1"))
        insert!(db, [-0.9, 0.1], (side = "left", name = "left-2"))
        insert!(db, [0.9, 0.1], (side = "right", name = "right-1"))
        insert!(db, [1.0, 0.0], (side = "right", name = "right-2"))

        build!(db; nlists = 2, iterations = 20, seed = 42)
        save!(db)
        snapshot_path=current_database_snapshot(path)

        @test isfile(joinpath(path, "CURRENT"))
        @test isfile(joinpath(snapshot_path, "manifest.toml"))
        @test isfile(joinpath(snapshot_path, "segments.toml"))
        @test !isfile(joinpath(snapshot_path, "vectors.bin"))
        @test !isfile(joinpath(snapshot_path, "metadata.bin"))
        @test !isfile(joinpath(snapshot_path, "ids.bin"))
        @test !isfile(joinpath(snapshot_path, "index.bin"))
        @test isfile(joinpath(snapshot_path, "segment-000001-vectors.bin"))
        @test isfile(joinpath(snapshot_path, "segment-000001-index.bin"))
        @test isfile(joinpath(snapshot_path, "snapshot.toml"))

        expected_length=length(db)
        close(db)
        loaded=load_db(path; iterations = 20, seed = 42)

        @test loaded.path==path
        @test loaded.dim==db.dim
        @test loaded.metric==db.metric
        @test length(loaded)==expected_length
        @test stored_vectors(loaded.vector_store)==stored_vectors(db.vector_store)
        @test stored_metadata(loaded.metadata_store)==stored_metadata(db.metadata_store)
        @test stored_ids(loaded.id_store)==stored_ids(db.id_store)
        @test loaded.index isa FilterAwareIVFIndex
        @test loaded.filter_index isa BitsetIndex
        @test loaded.index.ivf.centroids==db.index.ivf.centroids
        @test loaded.index.ivf.lists==db.index.ivf.lists
        @test loaded.index.ivf.list_radii==db.index.ivf.list_radii

        results=search(
            loaded,
            [-1.0, 0.0];
            k = 2,
            nprobe = 1,
            filter = (side = "right",),
            strategy = :filter_aware,
            vector_weight = 0.0,
            filter_weight = 1.0,
        )

        @test length(results)==2
        @test all(result->result.metadata.side=="right", results)
        @test Set(result.id for result in results)==Set([3, 4])
        close(loaded)
    end

    mktempdir() do path
        database_path=joinpath(path, "unbuilt")
        db=create_db(database_path; dim = 2)

        insert!(db, [1.0, 0.0], (name = "first",))
        save!(db)
        snapshot_path=current_database_snapshot(database_path)

        close(db)
        loaded=load_db(database_path)

        @test length(loaded)==1
        @test loaded.index===nothing
        @test loaded.filter_index===nothing
        @test !isfile(joinpath(snapshot_path, "index.bin"))
        close(loaded)
    end

    mktempdir() do path
        db=create_db(path; dim = 2)
        insert!(db, [1.0, 0.0], (name = "first",))
        save!(db)
        first_snapshot=current_database_snapshot(path)

        insert!(db, [0.0, 1.0], (name = "second",))
        save!(db)
        second_snapshot=current_database_snapshot(path)

        @test first_snapshot!=second_snapshot
        @test length(database_snapshot_generations(path))==2
        close(db)
        db=load_db(path)
        @test length(db)==2

        incomplete_path=joinpath(path, "snapshots", ".tmp-interrupted")
        mkpath(incomplete_path)
        write(joinpath(incomplete_path, "manifest.toml"), "incomplete")

        @test current_database_snapshot(path)==second_snapshot
        close(db)
        db=load_db(path)
        @test length(db)==2

        insert!(db, [-1.0, 0.0], (name = "third",))
        save!(db; retain_snapshots = 2)
        third_snapshot=current_database_snapshot(path)

        @test third_snapshot!=second_snapshot
        @test length(database_snapshot_generations(path))==2
        @test !isdir(first_snapshot)
        @test isdir(second_snapshot)
        @test isdir(third_snapshot)
        close(db)
        db=load_db(path)
        @test length(db)==3

        close(db)
        write(joinpath(path, "CURRENT"), basename(second_snapshot))

        @test current_database_snapshot(path)==second_snapshot
        db=load_db(path)
        @test length(db)==2
        close(db)
    end

    mktempdir() do path
        vector_store=create_vector_store(2)
        metadata_store=create_metadata_store()
        id_store=create_id_store()
        insert_vector!(vector_store, [1.0, 0.0])
        insert_metadata!(metadata_store, (name = "legacy",))
        insert_id!(id_store, 1)
        save_manifest(path, create_database_manifest(2, :cosine, 1))
        save_vector_store(path, vector_store)
        save_metadata_store(path, metadata_store)
        save_id_store(path, id_store)

        loaded=load_db(path)
        @test length(loaded)==1
        @test loaded.index===nothing
        close(loaded)
    end

    mktempdir() do path
        db=create_db(path; dim = 2)
        insert!(db, [1.0, 0.0], (name = "first",); id = "first")
        save!(db)
        first_snapshot=current_database_snapshot(path)
        insert!(db, [0.0, 1.0], (name = "second",); id = "second")
        save!(db)
        corrupted_snapshot=current_database_snapshot(path)

        open(joinpath(corrupted_snapshot, "vectors.bin"), "a") do io
            write(io, UInt8(0xff))
        end

        close(db)
        @test_throws ArgumentError load_db(path)

        recovered=recover_db(path)

        @test length(recovered)==1
        @test current_database_snapshot(path)==first_snapshot
        @test get_record(recovered, "first").metadata.name=="first"
        @test_throws KeyError get_record(recovered, "second")
        @test recovered.wal_checkpoint_revision==recovered.revision
        insert!(recovered, [-1.0, 0.0], (name = "after-recovery",); id = "after-recovery")
        save!(recovered)
        close(recovered)

        reopened=load_db(path)
        @test get_record(reopened, "after-recovery").metadata.name=="after-recovery"
        close(reopened)
    end

    mktempdir() do path
        db=create_db(path; dim = 2)
        insert!(db, [1.0, 0.0], (name = "first",); id = "first")
        build!(db; nlists = 1, iterations = 2)
        save!(db)
        first_snapshot=current_database_snapshot(path)

        insert!(db, [0.0, 1.0], (name = "second",); id = "second")
        save!(db)
        corrupted_snapshot=current_database_snapshot(path)

        @test isfile(joinpath(corrupted_snapshot, "segments.toml"))
        @test !isfile(joinpath(corrupted_snapshot, "vectors.bin"))
        open(joinpath(corrupted_snapshot, "segment-000001-vectors.bin"), "a") do io
            write(io, UInt8(0xff))
        end

        close(db)
        @test_throws ArgumentError load_db(path)

        recovered=recover_db(path)

        @test current_database_snapshot(path)==first_snapshot
        @test get_record(recovered, "first").metadata.name=="first"
        @test_throws KeyError get_record(recovered, "second")
        close(recovered)
    end

    mktempdir() do path
        db=create_db(path; dim = 2)
        insert!(db, [1.0, 0.0], (name = "first",))
        save!(db)
        insert!(db, [0.0, 1.0], (name = "second",))
        save!(db)
        latest_snapshot=current_database_snapshot(path)
        partial_path=joinpath(path, "snapshots", "9999999999999999999-1")
        mkpath(partial_path)
        write(joinpath(partial_path, "manifest.toml"), "incomplete")
        rm(joinpath(path, "CURRENT"))

        close(db)
        @test_throws ArgumentError load_db(path)

        recovered=recover_db(path)

        @test length(recovered)==2
        @test current_database_snapshot(path)==latest_snapshot
        @test isfile(joinpath(path, "CURRENT"))
        close(recovered)
    end

    mktempdir() do path
        db=create_db(path; dim = 2)
        insert!(db, [1.0, 0.0], (name = "right",); id = "right")
        insert!(db, [-1.0, 0.0], (name = "left",); id = "left")
        build!(db; nlists = 1, iterations = 2)
        save!(db)
        close(db)

        loaded=load_db(path; mmap_vectors = true)
        mapped_store=loaded.vector_store

        @test is_mapped(mapped_store)
        @test [
            result.id for result in search(loaded, [1.0, 0.0]; k = 2, strategy = :exact)
        ]==["right", "left"]

        save!(loaded; retain_snapshots = 1)
        @test is_mapped(mapped_store)
        @test [
            result.id for
            result in search(loaded, [-1.0, 0.0]; k = 1, strategy = :ivf, nprobe = 1)
        ]==["left"]

        close(loaded)
        @test !is_mapped(mapped_store)
        @test size(stored_vectors(mapped_store))==(2, 0)

        copied=load_db(path; mmap_vectors = :auto, mmap_threshold_bytes = typemax(Int))
        @test !is_mapped(copied.vector_store)
        @test get_record(copied, "right").metadata.name=="right"
        close(copied)
    end

    mktempdir() do path
        db=create_db(path; dim = 2)
        insert!(db, [1.0, 0.0], (version = 1,); id = "first")
        insert!(db, [0.0, 1.0], (version = 1,); id = "second")
        save!(db)
        close(db)

        loaded=load_db(path; mmap_vectors = true)
        mapped_store=loaded.vector_store

        @test is_mapped(mapped_store)
        @test !is_built(loaded)

        insert!(loaded, [-1.0, 0.0], (version = 1,); id = "third")
        @test !is_mapped(mapped_store)
        upsert!(loaded, [0.8, 0.2], (version = 2,); id = "first")
        update!(loaded, "second"; vector = [0.2, 0.8], metadata = (version = 2,))
        delete!(loaded, "third")

        @test length(loaded)==2
        @test get_record(loaded, "first").vector==Float32[0.8, 0.2]
        @test get_record(loaded, "second").metadata.version==2
        @test_throws KeyError get_record(loaded, "third")

        save!(loaded)
        expected_revision=loaded.revision
        close(loaded)
        reopened=load_db(path; mmap_vectors = true)

        @test reopened.revision==expected_revision
        @test get_record(reopened, "first").metadata.version==2
        @test get_record(reopened, "second").vector==Float32[0.2, 0.8]
        @test_throws KeyError get_record(reopened, "third")
        close(reopened)

        automatic=load_db(path; mmap_vectors = :auto, mmap_threshold_bytes = 0)
        @test is_mapped(automatic.vector_store)
        insert!(automatic, [-1.0, 0.0], (version = 1,); id = "automatic")
        upsert!(automatic, [0.7, 0.3], (version = 3,); id = "first")
        update!(automatic, "second"; metadata = (version = 3,))
        delete!(automatic, "automatic")
        @test get_record(automatic, "first").metadata.version==3
        @test get_record(automatic, "second").metadata.version==3
        @test_throws KeyError get_record(automatic, "automatic")
        close(automatic)
    end
end
