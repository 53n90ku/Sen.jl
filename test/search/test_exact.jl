using Test
using Sen
@testset "exact search" begin
    vectors = Float32[
        1 0 -1
        0 1 0
    ]
    metadata = [(name = "right",), (name = "up",), (name = "left",)]
    filter_index = build_bitset_index(metadata)
    query = Float32[1, 0]

    results = search_exact(vectors, metadata, query; k = 2, metric = :cosine)

    @test length(results)==2
    @test results[1].index==1
    @test results[1].score ≈ 1.0f0
    @test results[1].metadata.name=="right"
    @test results[2].index==2
    @test results[2].score ≈ 0.0f0

    filtered_results =
        search_exact(vectors, metadata, query; k = 2, filter = (name = "up",))
    @test length(filtered_results)==1
    @test filtered_results[1].index==2
    @test filtered_results[1].metadata.name=="up"

    empty_results =
        search_exact(vectors, metadata, query; k = 2, filter = (name = "missing",))
    @test isempty(empty_results)

    bitset_results=search_exact(
        vectors,
        metadata,
        query;
        k = 2,
        filter = (name = "up",),
        filter_index = filter_index,
    )
    @test length(bitset_results)==1
    @test bitset_results[1].index==2
    @test bitset_results[1].metadata.name=="up"

end
