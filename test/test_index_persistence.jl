using Test
using Sen

@testset "index persistence" begin
    dataset=generate_test_clusters(60,8,4;noise=0.05,seed=42,)
    index=build_ivf(dataset.vectors;nlists=4,iterations=10,seed=42,restarts=2,)

    mktempdir() do path
        index_path=save_ivf_index(path,index)
        loaded=load_ivf_index(path)

        @test isfile(index_path)
        @test loaded.centroids==index.centroids
        @test loaded.lists==index.lists
        @test loaded.vector_norms==index.vector_norms
        @test loaded.metric===index.metric
        @test loaded.list_radii==index.list_radii
        @test loaded.list_cos_radii==index.list_cos_radii
        @test loaded.list_sin_radii==index.list_sin_radii
    end

    mktempdir() do path
        @test_throws ArgumentError load_ivf_index(path)

        open(joinpath(path,"index.bin"),"w") do io
            write(io,zeros(UInt8,8))
        end

        @test_throws ArgumentError load_ivf_index(path)
    end
end
