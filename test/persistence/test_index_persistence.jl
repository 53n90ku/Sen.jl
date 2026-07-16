using Test
using Sen

@testset "index persistence" begin
    dataset=generate_test_clusters(60, 8, 4; noise = 0.05, seed = 42)
    index=build_ivf(dataset.vectors; nlists = 4, iterations = 10, seed = 42, restarts = 2)

    mktempdir() do path
        index_path=save_ivf_index(path, index)
        loaded=load_ivf_index(path)

        @test isfile(index_path)
        @test loaded.centroids==index.centroids
        @test loaded.lists==index.lists
        @test loaded.vector_norms==index.vector_norms
        @test loaded.metric===index.metric
        @test loaded.routing===index.routing
        @test loaded.list_radii==index.list_radii
        @test loaded.list_cos_radii==index.list_cos_radii
        @test loaded.list_sin_radii==index.list_sin_radii
    end

    mktempdir() do path
        @test_throws ArgumentError load_ivf_index(path)

        open(joinpath(path, "index.bin"), "w") do io
            write(io, zeros(UInt8, 8))
        end

        @test_throws ArgumentError load_ivf_index(path)
    end


    mktempdir() do path
        legacy=build_ivf(
            Float32[10 1; 0 1];
            nlists = 2,
            iterations = 2,
            seed = 8,
            metric = :dot,
        )
        open(joinpath(path, "index.bin"), "w") do io
            write(io, Sen.IVF_INDEX_MAGIC)
            write(io, Int64(1))
            write(io, UInt8(2))
            write(io, Int64(size(legacy.centroids, 1)))
            write(io, Int64(length(legacy.lists)))
            write(io, Int64(sum(length, legacy.lists)))
            write(io, legacy.centroids)
            write(io, legacy.vector_norms)
            write(io, legacy.list_radii)

            for list in legacy.lists
                write(io, Int64(length(list)))

                for vector_index in list
                    write(io, Int64(vector_index))
                end
            end
        end

        loaded=load_ivf_index(path)
        @test loaded.metric===:dot
        @test loaded.routing===:euclidean
    end
end
