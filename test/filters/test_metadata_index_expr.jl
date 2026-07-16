using Dates
using Test
using Sen
using Sen: BitsetIndex, build_bitset_index, evaluate_filter, matches_filter

@testset "adaptive metadata index expressions" begin
    metadata=NamedTuple[
        (group = index<=64 ? "common" : "other", tag = index) for index = 1:128
    ]
    index=build_bitset_index(metadata)

    @test index isa BitsetIndex
    @test index.count==length(metadata)
    @test index.selections[(:group, "common")] isa Sen.DenseStoredSelection
    @test index.selections[(:tag, 1)] isa Sen.SparseStoredSelection

    @test evaluate_filter(index, Eq(:group, "common"))==BitVector([
        position<=64 for position = 1:128
    ])
    @test evaluate_filter(index, In(:tag, [1, 3, 128]))==BitVector([
        position in (1, 3, 128) for position = 1:128
    ])
    @test evaluate_filter(index, And(Eq(:group, "common"), Not(In(:tag, [1, 2]))))==BitVector([
        3<=position<=64 for position = 1:128
    ])
    @test evaluate_filter(index, Or(Eq(:tag, 1), Eq(:tag, 128)))==BitVector([
        position in (1, 128) for position = 1:128
    ])
    @test evaluate_filter(index, (group = "other", tag = 128))==BitVector([
        position==128 for position = 1:128
    ])

    owned=evaluate_filter(index, Eq(:group, "common"))
    owned[1]=false
    @test evaluate_filter(index, Eq(:group, "common"))[1]

    @test evaluate_filter(index, Range(:tag, 1, 10))==BitVector([
        1<=position<=10 for position = 1:128
    ])
end

@testset "temporal equality keeps Date and DateTime distinct" begin
    date=Date(2026, 7, 15)
    datetime=DateTime(2026, 7, 15)
    metadata=NamedTuple[
        (created = date,),
        (created = datetime,),
        (created = Date(2026, 7, 16),),
    ]
    index=build_bitset_index(metadata)

    for filter in
        (Eq(:created, date), Eq(:created, datetime), In(:created, [date, datetime]))
        indexed=evaluate_filter(index, filter)
        scalar=BitVector(matches_filter(row, filter) for row in metadata)
        @test indexed==scalar
    end
end

@testset "metadata index missing values and empty expressions" begin
    metadata=NamedTuple[(value = missing,), (other = true,), (value = 1,)]
    index=build_bitset_index(metadata)

    @test evaluate_filter(index, Eq(:value, missing))==BitVector([true, false, false])
    @test evaluate_filter(index, In(:value, Any[]))==falses(3)
    @test evaluate_filter(index, And())==trues(3)
    @test evaluate_filter(index, Or())==falses(3)
    @test evaluate_filter(index, Not(Eq(:missing, 1)))==trues(3)
end

@testset "high cardinality metadata stays sparse" begin
    count=2_000
    index=build_bitset_index(NamedTuple[(unique_id = position,) for position = 1:count])

    @test all(selection->selection isa Sen.SparseStoredSelection, values(index.selections))
    @test sum(selection->length(selection.positions), values(index.selections))==count
end
