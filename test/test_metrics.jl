using Test
using rakon
@testset "Dot prod" begin
    a = Float32[1,2]
    b = Float32[3,4]

    @test dot_similarity(a,b)==11.0f0
    @test_throws DimensionMismatch dot_similarity(a,Float32[1])
end

@testset "Cosine similarity" begin
    x= Float32[1,0]
    same =Float32[1,0]
    perpendicular = Float32[0,1]
    opposite = Float32[-1,0]

    @test cosine_similarity(x, same) ≈ 1.0f0
    @test cosine_similarity(x, perpendicular) ≈ 0.0f0
    @test cosine_similarity(x, opposite) ≈ -1.0f0

    @test_throws ArgumentError cosine_similarity(
        Float32[0,0], Float32[1,0],
    )
end

@testset "Top k " begin
    results = top_k(Float32[0.2,0.9,0.4,0.7],2)

    @test length(results)==2
    @test results[1].index==2
    @test results[1].score==0.9f0
    @test results[2].index==4
    @test results[2].score ≈ 0.7f0

    @test_throws ArgumentError top_k(Float32[0.2,0.9],0)
    @test_throws ArgumentError top_k(Float32[0.2,0.9],3)
end
