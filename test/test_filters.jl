using Test
using rakon

@testset "metadata filters" begin
    metadata = (
        language="julia",topic="systems",year = 2025
    )

    @test matches_filter(metadata,(language="julia",))
    @test matches_filter(metadata,(language="julia",topic="systems"))

    @test !matches_filter(metadata,(language="python",))
    @test !matches_filter(metadata,(missing_field = "value",))

@testset "bitset metadata index" begin
    metadata = [
        (language="julia",topic="systems"),
        (language="python",topic="systems"),
        (language="julia",topic="databases"),
    ]
    index = build_bitset_index(metadata)
    julia_mask = evaluate_filter(index,(language="julia",),)
    @test julia_mask==BitVector([true,false,true])
    combined_mask = evaluate_filter(index, (language="julia",topic="systems"),)
    @test combined_mask==BitVector([true,false,false])
    missing_mask =evaluate_filter(index,(language="rust",),)
    @test missing_mask==BitVector([false,false,false])

@testset "filter selectivity" begin
    metadata=[
        (language="julia",topic="systems"),
        (language="python",topic="systems"),
        (language="julia",topic="databases"),
    ]
    index =build_bitset_index(metadata)
    julia_selectivity = estimate_selectivity(index,(language="julia",),)
    @test julia_selectivity≈2/3
    combined_selectivity = estimate_selectivity(index,(language="julia",topic="systems"),)
    @test combined_selectivity≈1/3
    missing_selectivity = estimate_selectivity(index,(language="rust",),)
    @test missing_selectivity==0.0

end
end
end