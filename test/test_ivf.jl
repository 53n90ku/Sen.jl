using Test
using Sen

@testset "sqared distance" begin
    a = Float32[1,2]
    b = Float32[4,6]

    @test squared_distance(a,b) == 25.0f0
    @test_throws DimensionMismatch squared_distance(a,Float32[1])
end

@testset "centroid training" begin
    vectors = Float32[
        -1.0 -0.9 0.9 1.0
        0.0 0.0 0.0 0.0
    ]
    centroids = train_centroids(vectors;nlists=2,iterations = 20,seed = 42,)

    @test size(centroids)==(2,2)

    left_distance = minimum(abs.(centroids[1,:].-(-0.95f0)),)
    right_distance = minimum(abs.(centroids[1,:].-0.95f0),)

    @test left_distance < 0.1f0
    @test right_distance < 0.1f0

    @test_throws ArgumentError train_centroids(vectors; nlists=5,)

@testset "build ivf index" begin
    vectors = Float32[
        -1.0 -0.9 0.9 1.0
        0.0 0.0 0.0 0.0
    ]
    index = build_ivf(vectors;nlists=2,iterations = 20,seed = 42,)

    @test size(index.centroids)==(2,2)
    @test length(index.lists)==2

    assigned_ids = sort(vcat(index.lists...))
    @test assigned_ids == [1,2,3,4]

    left_list = findfirst(list-> 1 in list ,index.lists)
    right_list = findfirst(list-> 3 in list, index.lists)

    @test left_list !== nothing
    @test right_list !==nothing
    @test 2 in index.lists[left_list]
    @test 4 in index.lists[right_list]
    @test left_list != right_list

    end

@testset "ivf search" begin
    vectors=Float32[
        -1.0 -0.9 0.9 1.0
        0.0 0.1 0.1 0.0
    ]
    metadata = [
        (name = "left-1",),
        (name = "left-2",),
        (name = "right-1",),
        (name = "right-2",),
    ]
    index = build_ivf(vectors; nlists=2,iterations=20,seed = 42,)
    query = Float32[-1,0]

    results = search_ivf(index,vectors,metadata,query;k=2,nprobe =1,)
    result_ids = [result.index for result in results]

    @test length(results)==2
    @test Set(result_ids)==Set([1,2])

    full_results = search_ivf(index,vectors,metadata,query;k=4,nprobe=2,)
    exact_results = search_exact(vectors,metadata,query;k=4,)

    full_ids = [result.index for result in full_results]
    exact_ids = [result.index for result in exact_results]

    @test recall_at_k(full_ids,exact_ids,4)==1.0
    @test_throws ArgumentError search_ivf(index,vectors,metadata,query;nprobe=0,)
    @test_throws ArgumentError search_ivf(index,vectors,metadata,query;nprobe=3,)

    end
end
