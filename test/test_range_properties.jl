using Dates
using Random
using Test
using Sen
using Sen: build_bitset_index, build_filter_aware_ivf, evaluate_filter
using Sen: estimate_list_filter_count, evaluate_list_filter, filtered_list_candidates

function range_property_metadata(rng::AbstractRNG, count::Int)
    numeric_values=Any[
        -Inf,
        -4,
        -1.5,
        -0.0,
        0.0,
        Float32(0.0),
        1,
        2.5,
        UInt64(4),
        Inf,
        NaN,
        missing,
    ]
    date_values=Any[Date(2023, 1, 1), Date(2024, 6, 1), Date(2025, 12, 31), missing]
    datetime_values=Any[
        DateTime(2023, 1, 1),
        DateTime(2024, 6, 1, 12),
        DateTime(2025, 12, 31, 23),
        missing,
    ]
    temporal_values=Any[date_values[1:3]..., datetime_values[1:3]..., missing]
    metadata=NamedTuple[]

    for index = 1:count
        group=rand(rng, ("a", "b", "c"))
        active=rand(rng, Bool)
        score=rand(rng, numeric_values)
        day=rand(rng, date_values)
        stamp=rand(rng, datetime_values)
        temporal=rand(rng, temporal_values)

        if index%11==0
            push!(
                metadata,
                (
                    group = group,
                    active = active,
                    day = day,
                    stamp = stamp,
                    temporal = temporal,
                ),
            )
        elseif index%13==0
            push!(
                metadata,
                (group = group, active = active, score = score, temporal = temporal),
            )
        else
            push!(
                metadata,
                (
                    group = group,
                    active = active,
                    score = score,
                    day = day,
                    stamp = stamp,
                    temporal = temporal,
                ),
            )
        end
    end

    return metadata
end

function random_range_property_atom(rng::AbstractRNG)
    choice=rand(rng, 1:10)

    choice==1&&return Sen.Range(:score, -2, 2)
    choice==2&&return Sen.Range(:score, Float32(-1.5), 2.5)
    choice==3&&return Sen.Range(:score, -0.0, 0.0)
    choice==4&&return Sen.Range(:score, -Inf, Inf)
    choice==5&&return Sen.Range(:day, Date(2023, 1, 1), Date(2025, 12, 31))
    choice==6&&return Sen.Range(
        :stamp,
        DateTime(2023, 1, 1),
        DateTime(2025, 12, 31, 23, 59),
    )
    choice==7&&return Sen.Range(:temporal, Date(2023, 1, 1), Date(2025, 12, 31))
    choice==8&&return Sen.Range(
        :temporal,
        DateTime(2023, 1, 1),
        DateTime(2025, 12, 31, 23, 59),
    )
    choice==9&&return Sen.Eq(:score, rand(rng, Any[NaN, -0.0, 0.0, missing, 1]))
    return Sen.In(:group, rand(rng, (["a"], ["b", "c"], String[])))
end

function random_range_property_expr(rng::AbstractRNG, depth::Int = 0)
    depth>=3&&return random_range_property_atom(rng)
    choice=rand(rng, 1:10)
    choice<=5&&return random_range_property_atom(rng)

    if choice<=7
        return Sen.And(
            random_range_property_expr(rng, depth+1),
            random_range_property_expr(rng, depth+1),
        )
    elseif choice<=9
        return Sen.Or(
            random_range_property_expr(rng, depth+1),
            random_range_property_expr(rng, depth+1),
        )
    end

    return Sen.Not(random_range_property_expr(rng, depth+1))
end

@testset "range index edge cases" begin
    metadata=[
        (value = -0.0, day = Date(2025, 1, 1), stamp = DateTime(2025, 1, 1)),
        (value = 0.0, day = Date(2025, 1, 2), stamp = DateTime(2025, 1, 2)),
        (value = NaN, day = DateTime(2025, 1, 1), stamp = Date(2025, 1, 1)),
        (value = missing, day = missing, stamp = missing),
        (other = 1,),
        (value = UInt64(2), day = Date(2025, 1, 3), stamp = DateTime(2025, 1, 3)),
        (value = Float32(2.5), day = Date(2025, 1, 4), stamp = DateTime(2025, 1, 4)),
    ]
    index=build_bitset_index(metadata)
    expressions=[
        Sen.Range(:value, -0.0, 0.0),
        Sen.Range(:value, -1, 2.5),
        Sen.Range(:value, -Inf, Inf),
        Sen.Range(:day, Date(2025, 1, 1), Date(2025, 1, 3)),
        Sen.Range(:stamp, DateTime(2025, 1, 1), DateTime(2025, 1, 3)),
        Sen.Not(Sen.Range(:value, -1, 1)),
        Sen.And(Sen.Range(:value, -1, 3), Sen.Not(Sen.Eq(:value, NaN))),
        Sen.Or(
            Sen.Range(:day, Date(2025, 1, 1), Date(2025, 1, 2)),
            Sen.Eq(:value, missing),
        ),
    ]

    for expression in expressions
        expected=BitVector(Sen.matches_filter(row, expression) for row in metadata)
        @test Sen.supports_indexed_filter(index, expression)
        @test evaluate_filter(index, expression)==expected
    end

    zero_mask=evaluate_filter(index, Sen.Range(:value, -0.0, 0.0))
    @test zero_mask[1]
    @test zero_mask[2]
    @test !zero_mask[3]
    @test !zero_mask[4]
    @test !zero_mask[5]

    date_mask=evaluate_filter(index, Sen.Range(:day, Date(2025, 1, 1), Date(2025, 1, 3)))
    @test date_mask==BitVector([true, true, false, false, false, true, false])
    datetime_mask=evaluate_filter(
        index,
        Sen.Range(:stamp, DateTime(2025, 1, 1), DateTime(2025, 1, 3)),
    )
    @test datetime_mask==BitVector([true, true, false, false, false, true, false])

    empty_index=build_bitset_index(NamedTuple[])

    for expression in expressions
        @test isempty(evaluate_filter(empty_index, expression))
    end
end

@testset "randomized range scalar index oracle" begin
    rng=MersenneTwister(117)
    metadata=range_property_metadata(rng, 80)
    index=build_bitset_index(metadata)
    expressions=[random_range_property_expr(rng) for _ = 1:60]

    for expression in expressions
        actual=evaluate_filter(index, expression)
        expected=BitVector(undef, length(metadata))

        for row_index in eachindex(metadata)
            expected[row_index]=Sen.matches_filter(metadata[row_index], expression)
            @test actual[row_index]==expected[row_index]
        end

        @test Sen.supports_indexed_filter(index, expression)
        @test actual==expected
    end
end

@testset "range list index mapping" begin
    rng=MersenneTwister(118)
    metadata=range_property_metadata(rng, 80)
    angles=range(0.01, 2pi-0.01; length = length(metadata))
    vectors=Matrix{Float32}(undef, 2, length(metadata))

    for (index, angle) in enumerate(angles)
        vectors[1, index]=Float32(cos(angle))
        vectors[2, index]=Float32(sin(angle))
    end

    index=build_filter_aware_ivf(
        vectors,
        metadata;
        nlists = 4,
        iterations = 4,
        seed = 118,
        metric = :cosine,
    )
    expressions=[random_range_property_expr(rng) for _ = 1:60]

    for expression in expressions
        for list_index in eachindex(index.ivf.lists)
            vector_indices=index.ivf.lists[list_index]
            expected=BitVector(
                Sen.matches_filter(metadata[vector_index], expression) for
                vector_index in vector_indices
            )
            mask=evaluate_list_filter(index, list_index, expression)
            candidates=filtered_list_candidates(index, list_index, expression)

            @test mask==expected
            @test candidates==Int[
                vector_indices[position] for
                position in eachindex(expected) if expected[position]
            ]
            @test estimate_list_filter_count(index, list_index, expression)==count(expected)
        end
    end
end
