using Test
using Sen
using Sen: build_bitset_index, estimate_selectivity
using Sen: build_filter_aware_ivf, search_exact
using Sen: search_filter_aware_bound, search_filter_aware_bound_with_stats

@testset "expression selectivity" begin
    metadata=[
        (group = "a", score = 1, active = true),
        (group = "a", score = 2, active = false),
        (group = "b", score = 3, active = true),
        (group = "b", score = 4, active = false),
        (group = "c", score = 5, active = true),
        (score = 6, active = false),
    ]
    index=build_bitset_index(metadata)
    expressions=[
        Sen.Eq(:group, "a"),
        Sen.In(:group, ["a", "c"]),
        Sen.And(Sen.In(:group, ["a", "b"]), Sen.Eq(:active, true)),
        Sen.Or(Sen.Eq(:group, "c"), Sen.Eq(:active, false)),
        Sen.Not(Sen.Eq(:group, "a")),
        Sen.And(
            Sen.Or(Sen.Eq(:group, "a"), Sen.Eq(:group, "c")),
            Sen.Not(Sen.Eq(:active, false)),
        ),
        Sen.And(),
        Sen.Or(),
    ]

    for expression in expressions
        expected=count(row->Sen.matches_filter(row, expression), metadata)/length(metadata)
        @test estimate_selectivity(index, expression)==expected
    end

    legacy=(group = "a", active = true)
    @test estimate_selectivity(index, legacy)==estimate_selectivity(
        index,
        Sen.normalize_filter(legacy),
    )

    range=Sen.Range(:score, 2, 4)
    @test estimate_selectivity(index, range)==0.5
    @test estimate_selectivity(index, range; metadata = metadata)==0.5

    nested_range=Sen.Or(Sen.Eq(:group, "c"), Sen.Range(:score, 2, 3))
    expected=count(row->Sen.matches_filter(row, nested_range), metadata)/length(metadata)
    @test estimate_selectivity(index, nested_range; metadata = metadata)==expected
    @test_throws DimensionMismatch estimate_selectivity(
        index,
        range;
        metadata = metadata[1:2],
    )

    empty_index=build_bitset_index(NamedTuple[])
    @test estimate_selectivity(empty_index, range)==0.0
end

@testset "bounded expression search" begin
    vectors=Float32[
        1.0 0.9 0.7 0.2 -0.2 -0.7 -0.9 -1.0
        0.0 0.4 0.7 1.0 1.0 0.7 0.4 0.0
    ]
    metadata=[
        (group = "a", active = true, score = 1),
        (group = "a", active = false, score = 2),
        (group = "b", active = true, score = 3),
        (group = "b", active = false, score = 4),
        (group = "c", active = true, score = 5),
        (group = "c", active = false, score = 6),
        (group = "d", active = true, score = 7),
        (group = "d", active = false, score = 8),
    ]
    query=Float32[1.0, 0.0]
    index=build_filter_aware_ivf(
        vectors,
        metadata;
        nlists = 2,
        iterations = 5,
        seed = 73,
        metric = :cosine,
    )
    expressions=[
        Sen.Eq(:group, "a"),
        Sen.In(:group, ["a", "c"]),
        Sen.And(Sen.In(:group, ["a", "b"]), Sen.Eq(:active, true)),
        Sen.Or(Sen.Eq(:group, "a"), Sen.Eq(:group, "d")),
        Sen.Not(Sen.Eq(:group, "a")),
        Sen.And(
            Sen.Or(Sen.Eq(:group, "a"), Sen.Eq(:group, "c")),
            Sen.Not(Sen.Eq(:active, false)),
        ),
        Sen.And(),
        Sen.Or(),
    ]

    for expression in expressions
        expected=search_exact(
            vectors,
            metadata,
            query;
            k = 8,
            metric = :cosine,
            filter = expression,
        )
        stats=search_filter_aware_bound_with_stats(
            index,
            vectors,
            metadata,
            query;
            filter = expression,
            k = 8,
            minimum_nprobe = 1,
            max_nprobe = length(index.ivf.lists),
            metric = :cosine,
        )
        actual=search_filter_aware_bound(
            index,
            vectors,
            metadata,
            query;
            filter = expression,
            k = 8,
            minimum_nprobe = 1,
            max_nprobe = length(index.ivf.lists),
            metric = :cosine,
        )

        @test stats.exact
        @test [result.index for result in stats.results]==[result.index for result in expected]
        @test [result.index for result in actual]==[result.index for result in expected]
        @test all(result->Sen.matches_filter(result.metadata, expression), actual)
    end

    legacy=(group = "a", active = true)
    legacy_results=search_filter_aware_bound(
        index,
        vectors,
        metadata,
        query;
        filter = legacy,
        k = 8,
        max_nprobe = length(index.ivf.lists),
    )
    expression_results=search_filter_aware_bound(
        index,
        vectors,
        metadata,
        query;
        filter = Sen.normalize_filter(legacy),
        k = 8,
        max_nprobe = length(index.ivf.lists),
    )
    @test [result.index for result in legacy_results]==[result.index for result in expression_results]

    range=Sen.Range(:score, 2, 4)
    range_expected=search_exact(
        vectors,
        metadata,
        query;
        k = 8,
        metric = :cosine,
        filter = range,
    )
    range_results=search_filter_aware_bound(
        index,
        vectors,
        metadata,
        query;
        filter = range,
        k = 8,
        max_nprobe = length(index.ivf.lists),
    )
    @test [result.index for result in range_results]==[result.index for result in range_expected]
end
