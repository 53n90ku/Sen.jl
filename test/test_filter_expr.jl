using Test
using Dates

module FilterExprContract
using Dates
include(joinpath(@__DIR__, "..", "src", "filters", "filter_expr.jl"))
end

using .FilterExprContract: FilterExpr, Eq, In, Range, And, Or, Not
using .FilterExprContract:
    normalize_filter, validate_filter, matches_filter, filter_value_isequal
using .FilterExprContract: MAX_FILTER_DEPTH, MAX_FILTER_NODES, MAX_FILTER_IN_VALUES

@testset "filter expression construction" begin
    eq=Eq(:language, SubString("julia", 1, 5))
    @test eq.field===:language
    @test eq.value=="julia"
    @test eq.value isa String

    membership=In(:language, ["julia", "rust", "julia"])
    @test membership.values==("julia", "rust")
    source=["julia", "rust"]
    frozen=In(:language, source)
    push!(source, "python")
    @test frozen.values==("julia", "rust")

    special=In(:value, [missing, missing, NaN, NaN, 1, 1.0, 0.0, -0.0])
    @test length(special.values)==5
    @test filter_value_isequal(special.values[1], missing)
    @test isnan(special.values[2])
    @test special.values[3]==1
    @test isequal(special.values[4], 0.0)
    @test isequal(special.values[5], -0.0)

    @test And().children===()
    @test Or().children===()
    @test Not(eq).child===eq
    @test_throws ArgumentError Eq("language", "julia")
    @test_throws ArgumentError In("language", ["julia"])
    @test_throws ArgumentError Range("year", 2020, 2030)
    @test_throws ArgumentError In(:language, "julia")
    @test_throws ArgumentError Eq(:tags, ["julia"])
    @test_throws ArgumentError And((eq, 1))
    @test_throws ArgumentError Not(1)
end

@testset "filter expression validation" begin
    @test normalize_filter(nothing)===nothing
    normalized=normalize_filter((language = "julia", year = 2026))
    @test normalized==And(Eq(:language, "julia"), Eq(:year, 2026))
    @test normalize_filter(NamedTuple())==And()

    depth_limit=Eq(:value, 1)
    for _ = 2:MAX_FILTER_DEPTH
        depth_limit=Not(depth_limit)
    end
    @test validate_filter(depth_limit)===depth_limit
    @test_throws ArgumentError validate_filter(Not(depth_limit))

    node_limit=And(Tuple(Eq(:value, index) for index = 1:(MAX_FILTER_NODES-1)))
    @test validate_filter(node_limit)===node_limit
    @test_throws ArgumentError validate_filter(And(node_limit, Eq(:value, 0)))
    @test_throws ArgumentError In(:value, 1:(MAX_FILTER_IN_VALUES+1))
    @test_throws ArgumentError In(:value, Iterators.repeated(1, MAX_FILTER_IN_VALUES+1))
end

@testset "filter scalar evaluation" begin
    metadata=(
        language = "julia",
        year = 2026,
        score = NaN,
        optional = missing,
        zero = 0.0,
        date = Date(2026, 7, 15),
        created_at = DateTime(2026, 7, 15, 10, 30),
    )

    @test matches_filter(metadata, Eq(:language, "julia"))
    @test !matches_filter(metadata, Eq(:missing_field, "julia"))
    @test matches_filter(metadata, Eq(:optional, missing))
    @test matches_filter(metadata, Eq(:score, NaN))
    @test !matches_filter(metadata, Eq(:zero, -0.0))
    @test matches_filter(metadata, In(:language, ["rust", "julia"]))
    @test !matches_filter(metadata, In(:language, []))
    @test matches_filter(metadata, In(:optional, [nothing, missing]))

    @test matches_filter(metadata, Range(:year, 2020, 2026))
    @test matches_filter(metadata, Range(:year, 2026.0, 2030.0))
    @test matches_filter(metadata, Range(:date, Date(2026, 1, 1), Date(2026, 12, 31)))
    @test matches_filter(
        metadata,
        Range(:created_at, DateTime(2026, 7, 15), DateTime(2026, 7, 16)),
    )
    @test !matches_filter(
        metadata,
        Range(:date, DateTime(2026, 1, 1), DateTime(2026, 12, 31)),
    )
    @test !matches_filter(metadata, Range(:optional, 1, 2))
    @test !matches_filter(metadata, Range(:score, 0.0, 1.0))

    @test matches_filter(metadata, And(Eq(:language, "julia"), Range(:year, 2026, 2026)))
    @test matches_filter(metadata, Or(Eq(:language, "rust"), Eq(:year, 2026)))
    @test matches_filter(metadata, Not(Eq(:absent, true)))
    @test matches_filter(metadata, (language = "julia", year = 2026))
    @test matches_filter(metadata, NamedTuple())
    @test !matches_filter(metadata, Or())
end

@testset "Range validation" begin
    @test_throws ArgumentError Range(:value, 2, 1)
    @test_throws ArgumentError Range(:value, NaN, 1.0)
    @test_throws ArgumentError Range(:value, 1.0, NaN)
    @test_throws ArgumentError Range(:value, missing, 1)
    @test_throws ArgumentError Range(:value, nothing, 1)
    @test_throws ArgumentError Range(:value, false, true)
    @test_throws ArgumentError Range(:value, Date(2026, 1, 1), DateTime(2026, 1, 2))
    @test Range(:value, -Inf, Inf).lower==-Inf
end

@testset "filter expression equality and hashing" begin
    expressions=[
        Eq(:value, NaN),
        In(:value, [missing, NaN, 1]),
        Range(:value, 1, 3),
        And(Eq(:a, 1), Not(Eq(:b, 2))),
        Or(Eq(:a, 1), Eq(:a, 2)),
    ]

    copies=[
        Eq(:value, NaN),
        In(:value, [missing, NaN, 1.0]),
        Range(:value, 1.0, 3.0),
        And(Eq(:a, 1.0), Not(Eq(:b, 2.0))),
        Or(Eq(:a, 1.0), Eq(:a, 2.0)),
    ]

    for (expression, copy) in zip(expressions, copies)
        @test expression==copy
        @test isequal(expression, copy)
        @test hash(expression)==hash(copy)
        @test Dict(expression=>1)[copy]==1
    end

    @test Eq(:date, Date(2026, 1, 1))!=Eq(:date, DateTime(2026, 1, 1))
    @test In(:date, [Date(2026, 1, 1)])!=In(:date, [DateTime(2026, 1, 1)])
    @test And(Eq(:a, 1), Eq(:b, 2))!=And(Eq(:b, 2), Eq(:a, 1))
    @test Eq(:a, 1)!=In(:a, [1])
end
