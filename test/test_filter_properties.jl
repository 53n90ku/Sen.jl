using Dates
using Random
using Test
using Sen

function filter_test_equal(left,right)
    temporal=left isa Union{Date,DateTime}||right isa Union{Date,DateTime}
    temporal&&typeof(left)!==typeof(right)&&return false
    return isequal(left,right)
end

function filter_test_range(value,lower,upper)
    value===missing&&return false

    temporal=value isa Union{Date,DateTime}||lower isa Union{Date,DateTime}||upper isa Union{Date,DateTime}
    temporal&&!(typeof(value)===typeof(lower)===typeof(upper))&&return false

    try
        return lower<=value<=upper
    catch
        return false
    end
end

abstract type FilterTestExpr end

struct FilterTestEq <: FilterTestExpr
    field::Symbol
    value
end

struct FilterTestIn <: FilterTestExpr
    field::Symbol
    values::Tuple
end

struct FilterTestRange <: FilterTestExpr
    field::Symbol
    lower
    upper
end

struct FilterTestAnd <: FilterTestExpr
    children::Tuple
end

struct FilterTestOr <: FilterTestExpr
    children::Tuple
end

struct FilterTestNot <: FilterTestExpr
    child::FilterTestExpr
end

function filter_test_matches(row::NamedTuple,expr::FilterTestEq)
    hasproperty(row,expr.field)||return false
    return filter_test_equal(getproperty(row,expr.field),expr.value)
end

function filter_test_matches(row::NamedTuple,expr::FilterTestIn)
    hasproperty(row,expr.field)||return false
    value=getproperty(row,expr.field)
    return any(expected->filter_test_equal(value,expected),expr.values)
end

function filter_test_matches(row::NamedTuple,expr::FilterTestRange)
    hasproperty(row,expr.field)||return false
    return filter_test_range(getproperty(row,expr.field),expr.lower,expr.upper)
end

filter_test_matches(row::NamedTuple,expr::FilterTestAnd)=all(child->filter_test_matches(row,child),expr.children)
filter_test_matches(row::NamedTuple,expr::FilterTestOr)=any(child->filter_test_matches(row,child),expr.children)
filter_test_matches(row::NamedTuple,expr::FilterTestNot)=!filter_test_matches(row,expr.child)

function random_filter_atom(rng::AbstractRNG)
    choice=rand(rng,1:9)

    if choice==1
        value=rand(rng,(-2,-1,0,1,2,3))
        return(Sen.Eq(:integer,value),FilterTestEq(:integer,value))
    elseif choice==2
        values=Tuple(rand(rng,(-0.0,0.0,1.5,NaN,missing),rand(rng,0:4)))
        return(Sen.In(:floating,collect(values)),FilterTestIn(:floating,values))
    elseif choice==3
        lower=rand(rng,(-2.5,-1.0,0.0,1.5))
        upper=rand(rng,(2.0,3.5,5.0))
        return(Sen.Range(:integer,lower,upper),FilterTestRange(:integer,lower,upper))
    elseif choice==4
        lower=rand(rng,(-2,-1,0))
        upper=rand(rng,(1,2,4))
        return(Sen.Range(:floating,lower,upper),FilterTestRange(:floating,lower,upper))
    elseif choice==5
        values=Tuple(rand(rng,("a","b","c",missing),rand(rng,0:4)))
        return(Sen.In(:group,collect(values)),FilterTestIn(:group,values))
    elseif choice==6
        value=rand(rng,(true,false,missing))
        return(Sen.Eq(:optional,value),FilterTestEq(:optional,value))
    elseif choice==7
        lower=Date(2024,1,1)
        upper=Date(2026,1,1)
        return(Sen.Range(:date,lower,upper),FilterTestRange(:date,lower,upper))
    elseif choice==8
        lower=DateTime(2024,1,1)
        upper=DateTime(2026,1,1)
        return(Sen.Range(:datetime,lower,upper),FilterTestRange(:datetime,lower,upper))
    end

    value=rand(rng,(1,"missing",missing))
    return(Sen.Eq(:absent,value),FilterTestEq(:absent,value))
end

function random_filter_expr(rng::AbstractRNG,depth::Int=0)
    depth>=4&&return random_filter_atom(rng)
    choice=rand(rng,1:10)
    choice<=5&&return random_filter_atom(rng)

    if choice<=7
        count=rand(rng,0:3)
        pairs=[random_filter_expr(rng,depth+1) for _ in 1:count]
        return(Sen.And((pair[1] for pair in pairs)...),FilterTestAnd(Tuple(pair[2] for pair in pairs)))
    elseif choice<=9
        count=rand(rng,0:3)
        pairs=[random_filter_expr(rng,depth+1) for _ in 1:count]
        return(Sen.Or((pair[1] for pair in pairs)...),FilterTestOr(Tuple(pair[2] for pair in pairs)))
    end

    pair=random_filter_expr(rng,depth+1)
    return(Sen.Not(pair[1]),FilterTestNot(pair[2]))
end

@testset "filter expression scalar semantics" begin
    row=(
        integer=2,
        floating=-0.0,
        optional=missing,
        nan=NaN,
        date=Date(2025,1,1),
        datetime=DateTime(2025,1,1),
    )

    @test Sen.matches_filter(row,Sen.Eq(:integer,2))
    @test Sen.matches_filter(row,Sen.Eq(:integer,2.0))
    @test Sen.matches_filter(row,Sen.Eq(:optional,missing))
    @test Sen.matches_filter(row,Sen.Eq(:nan,NaN))
    @test Sen.matches_filter(row,Sen.Eq(:floating,-0.0))
    @test !Sen.matches_filter(row,Sen.Eq(:floating,0.0))
    @test !Sen.matches_filter(row,Sen.Eq(:absent,missing))
    @test !Sen.matches_filter(row,Sen.Eq(:date,DateTime(2025,1,1)))
    @test !Sen.matches_filter(row,Sen.Eq(:datetime,Date(2025,1,1)))

    @test Sen.matches_filter(row,Sen.In(:integer,[1,2,3]))
    @test Sen.matches_filter(row,Sen.In(:optional,[1,missing]))
    @test Sen.matches_filter(row,Sen.In(:nan,[0.0,NaN]))
    @test Sen.matches_filter(row,Sen.In(:floating,[-0.0]))
    @test !Sen.matches_filter(row,Sen.In(:floating,[0.0]))
    @test !Sen.matches_filter(row,Sen.In(:integer,Int[]))
    @test !Sen.matches_filter(row,Sen.In(:absent,[missing]))

    @test Sen.matches_filter(row,Sen.Range(:integer,1.5,2.5))
    @test Sen.matches_filter(row,Sen.Range(:integer,2,2))
    @test Sen.matches_filter(row,Sen.Range(:floating,-0.0,0.0))
    @test !Sen.matches_filter(row,Sen.Range(:nan,-1.0,1.0))
    @test !Sen.matches_filter(row,Sen.Range(:optional,0,1))
    @test !Sen.matches_filter(row,Sen.Range(:absent,0,1))
    @test !Sen.matches_filter(row,Sen.Range(:date,DateTime(2024),DateTime(2026)))
    @test !Sen.matches_filter(row,Sen.Range(:datetime,Date(2024),Date(2026)))
    @test_throws ArgumentError Sen.Range(:integer,3,2)
    @test_throws ArgumentError Sen.Range(:integer,NaN,2.0)
    @test_throws ArgumentError Sen.Range(:date,Date(2024),DateTime(2026))
end

@testset "filter expression boolean semantics" begin
    row=(group="a",score=3,active=true,optional=missing,)

    @test Sen.matches_filter(row,Sen.And())
    @test !Sen.matches_filter(row,Sen.Or())
    @test Sen.matches_filter(row,Sen.Not(Sen.Or()))
    @test !Sen.matches_filter(row,Sen.Not(Sen.And()))
    @test Sen.matches_filter(row,Sen.Not(Sen.Eq(:absent,1)))
    @test !Sen.matches_filter(row,Sen.Not(Sen.Eq(:optional,missing)))

    expr=Sen.And(
        Sen.Eq(:group,"a"),
        Sen.Or(
            Sen.Range(:score,2,4),
            Sen.Eq(:active,false),
        ),
        Sen.Not(Sen.In(:group,["b","c"])),
    )

    @test Sen.matches_filter(row,expr)
    @test !Sen.matches_filter(merge(row,(score=9,)),expr)
end

@testset "named tuple filter compatibility" begin
    filters=[
        NamedTuple(),
        (group="a",),
        (group="a",active=true,),
        (optional=missing,),
        (absent=missing,),
    ]
    rows=[
        (group="a",active=true,optional=missing,),
        (group="a",active=false,optional=1,),
        (group="b",active=true,optional=missing,),
    ]

    for filter in filters
        normalized=Sen.normalize_filter(filter)

        for row in rows
            @test Sen.matches_filter(row,filter)==Sen.matches_filter(row,normalized)
        end
    end
end

@testset "filter expression identity stability" begin
    left=Sen.And(Sen.Eq(:group,"a"),Sen.Or(Sen.Eq(:score,1),Sen.Not(Sen.Eq(:active,false))))
    right=Sen.And(Sen.Eq(:group,"a"),Sen.Or(Sen.Eq(:score,1),Sen.Not(Sen.Eq(:active,false))))

    @test left==right
    @test isequal(left,right)
    @test hash(left)==hash(right)
    @test Sen.normalize_filter((group="a",score=1,))==Sen.And(Sen.Eq(:group,"a"),Sen.Eq(:score,1))

    values=Any[1,NaN,-0.0]
    expression=Sen.In(:value,values)
    saved_hash=hash(expression)
    values[1]=99
    push!(values,100)

    @test hash(expression)==saved_hash
    @test Sen.matches_filter((value=1,),expression)
    @test Sen.matches_filter((value=NaN,),expression)
    @test Sen.matches_filter((value=-0.0,),expression)
    @test !Sen.matches_filter((value=99,),expression)
end

@testset "randomized filter expression oracle" begin
    rng=MersenneTwister(91)
    rows=[(
        integer=rand(rng,-3:3),
        floating=rand(rng,(-0.0,0.0,-1.5,1.5,NaN)),
        group=rand(rng,("a","b","c")),
        active=rand(rng,Bool),
        optional=rand(rng,(missing,missing,0,1)),
        date=Date(2023+rand(rng,0:4),1,1),
        datetime=DateTime(2023+rand(rng,0:4),1,1),
    ) for _ in 1:100]

    for _ in 1:300
        expression,oracle=random_filter_expr(rng)

        for row in rows
            @test Sen.matches_filter(row,expression)==filter_test_matches(row,oracle)
        end
    end
end

@testset "filter expression complexity limits" begin
    @test_throws ArgumentError Sen.normalize_filter(Sen.In(:value,collect(1:100_000)))

    @test_throws ArgumentError begin
        expression=Sen.Eq(:value,1)

        for _ in 1:512
            expression=Sen.Not(expression)
        end

        Sen.normalize_filter(expression)
    end

    @test_throws ArgumentError begin
        children=Tuple(Sen.Eq(:value,index) for index in 1:10_000)
        Sen.normalize_filter(Sen.And(children...))
    end
end
