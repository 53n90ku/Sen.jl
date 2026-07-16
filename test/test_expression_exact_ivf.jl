using Test
using Sen

using Sen: IVFIndex,build_bitset_index,filter_ivf_candidates,search_exact
using Sen: search_ivf_postfilter,search_ivf_prefilter

@testset "expression exact and IVF filtering" begin
    vectors=Float32[
        1.0 0.9 0.8 0.7 0.6
        0.0 0.0 0.0 0.0 0.0
    ]
    metadata=[
        (group="a",year=2020,active=true),
        (group="b",year=2021,active=true),
        (group="a",year=2022,active=false),
        (group="c",year=2023,active=true),
        (group="b",year=2024,active=false),
    ]
    query=Float32[1.0,0.0]
    ivf=IVFIndex(reshape(Float32[1.0,0.0],2,1),[collect(1:5)])
    metadata_index=build_bitset_index(metadata)

    boolean_filter=And(In(:group,["a","b"]),Not(Eq(:active,false)))
    scalar_exact=search_exact(vectors,metadata,query;k=5,metric=:dot,filter=boolean_filter,)
    indexed_exact=search_exact(vectors,metadata,query;k=5,metric=:dot,filter=boolean_filter,filter_index=metadata_index,)
    legacy_exact=search_exact(vectors,metadata,query;k=5,metric=:dot,filter=(group="a",),filter_index=metadata_index,)
    @test [result.index for result in scalar_exact]==[1,2]
    @test indexed_exact==scalar_exact
    @test [result.index for result in legacy_exact]==[1,3]

    range_filter=And(In(:group,["a","b"]),Range(:year,2021,2023))
    scalar_range=search_exact(vectors,metadata,query;k=5,metric=:dot,filter=range_filter,)
    indexed_range=search_exact(vectors,metadata,query;k=5,metric=:dot,filter=range_filter,filter_index=metadata_index,)
    @test [result.index for result in scalar_range]==[2,3]
    @test indexed_range==scalar_range

    candidates=filter_ivf_candidates(collect(1:5),metadata,range_filter;filter_index=metadata_index,)
    @test candidates==[2,3]

    prefiltered=search_ivf_prefilter(ivf,vectors,metadata,query;k=5,nprobe=1,metric=:dot,filter=range_filter,filter_index=metadata_index,)
    postfiltered=search_ivf_postfilter(ivf,vectors,metadata,query;k=5,nprobe=1,metric=:dot,filter=range_filter,oversample=1,)
    @test [result.index for result in prefiltered]==[2,3]
    @test [result.index for result in postfiltered]==[2,3]

    unfiltered=search_exact(vectors,metadata,query;k=5,metric=:dot,filter_index=build_bitset_index(metadata[1:4]),)
    @test [result.index for result in unfiltered]==collect(1:5)
    @test_throws DimensionMismatch search_exact(vectors,metadata,query;k=5,metric=:dot,filter=range_filter,filter_index=build_bitset_index(metadata[1:4]),)
    @test_throws DimensionMismatch filter_ivf_candidates(collect(1:5),metadata,range_filter;filter_index=build_bitset_index(metadata[1:4]),)
end
