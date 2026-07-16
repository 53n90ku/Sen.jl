using Dates
using Test
using Sen
using Sen: SparseStoredSelection,build_bitset_index,estimate_selectivity,evaluate_filter
using Sen: matches_filter,supports_indexed_filter

function range_oracle(metadata,filter)
    return BitVector(matches_filter(row,filter) for row in metadata)
end

@testset "ordered numeric range lanes match scalar semantics" begin
    large=UInt64(9_007_199_254_740_993)
    metadata=NamedTuple[
        (value=Int8(-2),zero=-0.0),
        (value=UInt16(0),zero=0.0),
        (value=Float32(1.5),zero=1.0),
        (value=2.0,zero=missing),
        (value=large,zero=NaN),
        (value=missing,zero=true),
        (value=NaN,zero="not numeric"),
        (value=true,zero=nothing),
        (other=4,zero=Inf),
    ]
    index=build_bitset_index(metadata)
    filters=(
        Range(:value,-2,2),
        Range(:value,0.0,1.5),
        Range(:value,2,2),
        Range(:value,large,large),
        Range(:zero,0.0,0.0),
        Range(:zero,-Inf,Inf),
        And(Range(:value,-2,2),Not(Eq(:value,0))),
        Or(Range(:value,large,large),Eq(:other,4)),
        Not(Range(:value,0,2)),
    )

    for filter in filters
        @test supports_indexed_filter(index,filter)
        @test evaluate_filter(index,filter)==range_oracle(metadata,filter)
    end

    @test evaluate_filter(index,Range(:absent,1,2))==falses(length(metadata))
    @test supports_indexed_filter(index,Range(:absent,1,2))
    @test estimate_selectivity(index,Range(:value,-2,2))≈4/length(metadata)
    @test sum(length(lane.values) for lanes in values(index.range_lanes) for lane in lanes)==10
    @test all(lane->eltype(lane.values)!==Any,Iterators.flatten(values(index.range_lanes)))
end

@testset "Date and DateTime range lanes stay strict" begin
    date=Date(2026,7,15)
    datetime=DateTime(2026,7,15)
    metadata=NamedTuple[
        (created=date,),
        (created=datetime,),
        (created=Date(2026,7,16),),
        (created=DateTime(2026,7,16),),
        (created=missing,),
    ]
    index=build_bitset_index(metadata)
    date_filter=Range(:created,date,Date(2026,7,16))
    datetime_filter=Range(:created,datetime,DateTime(2026,7,15,23,59))

    @test evaluate_filter(index,date_filter)==range_oracle(metadata,date_filter)==BitVector([true,false,true,false,false])
    @test evaluate_filter(index,datetime_filter)==range_oracle(metadata,datetime_filter)==BitVector([false,true,false,false,false])
    @test haskey(index.range_lanes,(:created,:date))
    @test haskey(index.range_lanes,(:created,:datetime))
end

@testset "high cardinality ranges remain linear" begin
    count=10_000
    index=build_bitset_index(NamedTuple[(number=position,) for position in 1:count])
    lanes=index.range_lanes[(:number,:numeric)]

    @test length(lanes)==1
    @test length(only(lanes).values)==count
    @test length(only(lanes).positions)==count
    @test all(selection->selection isa SparseStoredSelection,values(index.selections))
    @test evaluate_filter(index,Range(:number,4_500,5_500))==BitVector([4_500<=position<=5_500 for position in 1:count])
end
