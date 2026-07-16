using Dates
using Random
using Test
using Sen
using SenBench

@testset "expression workload generation" begin
    rng=MersenneTwister(211)
    vectors=randn(rng, Float32, 8, 1_000)
    workload=generate_expression_workload(vectors; workload = :random, seed = 212)
    repeated=generate_expression_workload(vectors; workload = :random, seed = 212)

    @test workload.metadata==repeated.metadata
    @test length(workload.metadata)==1_000
    @test length(workload.cases)==length(EXPRESSION_FAMILIES)*length(
        keys(EXPRESSION_BUCKETS),
    )
    @test Set(case.family for case in workload.cases)==Set(EXPRESSION_FAMILIES)
    @test Set(case.bucket for case in workload.cases)==Set(keys(EXPRESSION_BUCKETS))
    @test all(case->case.filter isa FilterExpr, workload.cases)

    index=Sen.build_bitset_index(workload.metadata)

    for case in workload.cases
        expected=Float64(getproperty(EXPRESSION_BUCKETS, case.bucket))
        mask=Sen.evaluate_filter(index, case.filter)
        scalar=BitVector(Sen.matches_filter(row, case.filter) for row in workload.metadata)
        @test mask==scalar
        @test case.actual_selectivity==count(mask)/length(mask)
        @test isapprox(case.actual_selectivity, expected; atol = 0.01)
    end

    @test_throws ArgumentError build_expression_filter(:missing, :rare, 1_000)
    @test_throws ArgumentError build_expression_filter(:eq, :missing, 1_000)
end

@testset "expression benchmark compatibility" begin
    rng=MersenneTwister(213)
    vectors=randn(rng, Float32, 6, 200)

    for column in axes(vectors, 2)
        vectors[:, column]./=sqrt(sum(abs2, @view vectors[:, column]))
    end

    workload=generate_expression_workload(
        vectors;
        workload = :correlated,
        seed = 214,
        families = (:range_numeric, :and, :not),
        buckets = (:medium,),
    )
    queries=vectors[:, 1:2]

    for case in workload.cases
        filters=fill(case.filter, size(queries, 2))
        truth=compute_groundtruth(
            vectors,
            workload.metadata,
            queries;
            k = 5,
            filters = filters,
        )
        result=run_benchmark(
            vectors,
            workload.metadata,
            queries,
            filters;
            nlists = 4,
            k = 5,
            nprobe = 4,
            iterations = 3,
            seed = 215,
            repetitions = 1,
        )

        @test length(truth)==2
        @test result.exact.average_recall==1.0
        @test result.ivf_prefilter.average_recall==1.0
        @test result.ivf_postfilter.average_recall==1.0
        @test result.filter_aware.average_recall==1.0
        @test result.filter_aware_bound.average_recall==1.0
    end
end
