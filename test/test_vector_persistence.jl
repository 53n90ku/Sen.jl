using Test
using Sen

@testset "vector persistence" begin
    store=create_vector_store(3; initial_capacity = 1)

    insert_vector!(store, [1.0, 2.0, 3.0])
    insert_vector!(store, [4.0, 5.0, 6.0])

    mktempdir() do path
        vector_path=save_vector_store(path, store)

        @test isfile(vector_path)

        loaded=load_vector_store(path; mmap = false)

        @test loaded.dim==store.dim
        @test loaded.count==store.count
        @test !is_mapped(loaded)
        @test size(stored_vectors(loaded))==(3, 2)
        @test stored_vectors(loaded)==stored_vectors(store)

        mapped=load_vector_store(path; mmap = true)

        @test is_mapped(mapped)
        @test stored_vectors(mapped)==stored_vectors(store)
        @test_throws ArgumentError insert_vector!(mapped, [7.0, 8.0, 9.0])
        @test_throws ArgumentError update_vector!(mapped, 1, [7.0, 8.0, 9.0])
        @test_throws ArgumentError swap_delete_vector!(mapped, 1)

        automatic=load_vector_store(path; mmap = :auto, mmap_threshold_bytes = 0)
        @test is_mapped(automatic)
        @test_throws ArgumentError load_vector_store(path; mmap = :invalid)
        @test_throws ArgumentError load_vector_store(
            path;
            mmap = :auto,
            mmap_threshold_bytes = -1,
        )
    end

    empty_store=create_vector_store(3)

    mktempdir() do path
        save_vector_store(path, empty_store)
        loaded=load_vector_store(path)

        @test loaded.dim==3
        @test loaded.count==0
        @test size(stored_vectors(loaded))==(3, 0)
    end

    mktempdir() do path
        @test_throws ArgumentError load_vector_store(path)
    end

    mktempdir() do path
        save_vector_store(path, store)
        vector_path=joinpath(path, "vectors.bin")

        open(vector_path, "r+") do io
            truncate(io, filesize(vector_path)-1)
        end

        @test_throws ArgumentError load_vector_store(path; mmap = true)
        @test_throws ArgumentError load_vector_store(path; mmap = false)
    end
end
