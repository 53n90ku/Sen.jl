using Test
using Dates
using Sen

using Sen: IVFIndex,build_filter_aware_ivf,collect_filtered_list_candidates
using Sen: estimate_list_filter_count,estimate_list_filter_density,evaluate_list_filter
using Sen: filtered_list_candidates,matches_filter,search_filter_aware_ivf

@testset "filter-aware list ranges" begin
    vectors=Float32[
        1.0 0.9 0.8 0.7 0.6 0.5
        0.0 0.0 0.0 0.0 0.0 0.0
    ]
    metadata=NamedTuple[
        (group="a",score=1,created=Date(2026,1,1)),
        (group="b",score=2,created=Date(2026,1,2)),
        (group="a",score=3,created=Date(2026,1,3)),
        (group="c",score=4,created=Date(2026,1,4)),
        (group="b",score=5,created=Date(2026,1,5)),
        (group="a",score=6,created=Date(2026,1,6)),
    ]
    ivf=IVFIndex(reshape(Float32[1.0,0.0],2,1),[[4,1,6,2,5,3]])
    index=build_filter_aware_ivf(ivf,metadata)
    query=Float32[1.0,0.0]
    filters=(
        Range(:score,2,5),
        And(In(:group,["a","b"]),Range(:score,2,5)),
        Or(Eq(:group,"c"),Range(:created,Date(2026,1,2),Date(2026,1,3))),
        Not(Range(:score,3,4)),
    )

    for filter in filters
        expected_mask=BitVector(matches_filter(metadata[global_id],filter) for global_id in ivf.lists[1])
        expected_ids=Int[ivf.lists[1][position] for position in eachindex(expected_mask) if expected_mask[position]]
        @test evaluate_list_filter(index,1,filter)==expected_mask
        @test estimate_list_filter_count(index,1,filter)==count(expected_mask)
        @test estimate_list_filter_density(index,1,filter)==count(expected_mask)/length(expected_mask)
        @test filtered_list_candidates(index,1,filter)==expected_ids
        @test collect_filtered_list_candidates(index,[1],filter)==expected_ids

        results=search_filter_aware_ivf(index,vectors,metadata,query;k=6,nprobe=1,metric=:dot,filter=filter,)
        @test Set(result.index for result in results)==Set(expected_ids)
    end
end

@testset "filter-aware absent and incompatible range lanes" begin
    vectors=Float32[1.0 0.9 0.8;0.0 0.0 0.0]
    metadata=NamedTuple[(score=1,label="a"),(score=2,label="b"),(score=3,label="c")]
    ivf=IVFIndex(reshape(Float32[1.0,0.0],2,1),[[3,1,2]])
    index=build_filter_aware_ivf(ivf,metadata)

    @test evaluate_list_filter(index,1,Range(:absent,1,3))==falses(3)
    @test evaluate_list_filter(index,1,Range(:label,1,3))==falses(3)
    @test evaluate_list_filter(index,1,Not(Range(:label,1,3)))==trues(3)
    @test filtered_list_candidates(index,1,And(Eq(:label,"a"),Range(:absent,1,3)))==Int[]
end
