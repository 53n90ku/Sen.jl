using Test
using Sen

@testset "synthetic dataset" begin
    dataset = generate_synthetic_dataset(100,16;seed = 42)

    @test size(dataset.vectors)==(16,100)
    @test length(dataset.metadata)==100
    
    for i in 1:100
        vector = @view dataset.vectors[:,i]
        @test isapprox(sum(abs2,vector),1.0f0;atol = 1.0f-5)
    end

    @test dataset.metadata[1].topic in (
        "systems","machine-learning","databases",
    )

@testset "recall at k" begin
    truth = [1,2,3,4]

    @test recall_at_k([1,2,3,4],truth,4)==1.0
    @test recall_at_k([1,3,8,9],truth,4)==0.5
    @test recall_at_k([8,9,10,11],truth,4)==0.0

    @test recall_at_k([1,2],truth,2)==1.0
    @test recall_at_k(Int[],Int[],4)==1.0
    @test recall_at_k([1],Int[],4)==0.0
    @test_throws ArgumentError recall_at_k([1],[1],0)
    end

@testset "exact ground truth" begin
    vectors = Float32[
        1 0 -1
        0 1 0
    ]
    metadata=[
        (name="right",),
        (name="up",),
        (name="left",),
    ]
    queries = Float32[
        1 0
        0 1
    ]
    truth = compute_groundtruth(vectors,metadata,queries;k=1,)
    @test length(truth)==2
    @test truth[1]==[1]
    @test truth[2]==[2]

    filtered_truth = compute_groundtruth(vectors,metadata,queries[:,1:1];k=1,filters=[(name="up",)],)
    @test filtered_truth[1]==[2]

    end
end
