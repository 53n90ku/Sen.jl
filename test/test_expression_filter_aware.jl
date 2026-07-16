using Test
using Sen
using Sen: FilterAwareIVFIndex,IVFIndex,build_filter_aware_ivf
using Sen: collect_filtered_list_candidates,estimate_list_filter_count
using Sen: estimate_list_filter_density,evaluate_list_filter,filtered_list_candidates
using Sen: matches_filter,search_filter_aware_ivf

@testset "filter aware metadata indexes evaluate expressions" begin
    vectors=Float32[
        1.0 0.9 0.0 -1.0 -0.9 0.0
        0.0 0.1 1.0 0.0 0.1 -1.0
    ]
    metadata=NamedTuple[
        (group="red",tag=1,active=true),
        (group="blue",tag=2,active=true),
        (group="red",tag=3,active=false),
        (group="green",tag=4,active=true),
        (group="blue",tag=5,active=false),
        (group="red",tag=6,active=true),
    ]
    ivf=IVFIndex(Float32[1.0 -1.0;0.0 0.0],[[3,1,5],[2,6,4]])
    index=build_filter_aware_ivf(ivf,metadata)
    filters=(
        Eq(:group,"red"),
        In(:tag,[1,4,6]),
        And(In(:group,["red","blue"]),Not(Eq(:active,false))),
        Or(Eq(:tag,3),Eq(:group,"green")),
        (group="blue",active=true,),
    )

    @test index isa FilterAwareIVFIndex
    @test length(index.metadata_indexes)==length(ivf.lists)
    @test all(list_index->index.metadata_indexes[list_index].count==length(ivf.lists[list_index]),eachindex(ivf.lists))

    for filter in filters
        for list_index in eachindex(ivf.lists)
            list=ivf.lists[list_index]
            expected_mask=BitVector(matches_filter(metadata[vector_index],filter) for vector_index in list)
            expected_candidates=Int[list[position] for position in eachindex(list) if expected_mask[position]]

            @test evaluate_list_filter(index,list_index,filter)==expected_mask
            @test estimate_list_filter_count(index,list_index,filter)==count(expected_mask)
            @test estimate_list_filter_density(index,list_index,filter)==count(expected_mask)/length(list)
            @test filtered_list_candidates(index,list_index,filter)==expected_candidates
        end

        expected=vcat((filtered_list_candidates(index,list_index,filter) for list_index in (2,1))...)
        @test collect_filtered_list_candidates(index,[2,1],filter)==expected
    end

    filter=And(In(:group,["red","green"]),Not(Eq(:active,false)))
    results=search_filter_aware_ivf(index,vectors,metadata,Float32[1.0,0.0];k=6,nprobe=2,filter=filter)
    @test Set(result.index for result in results)==Set(index for index in eachindex(metadata) if matches_filter(metadata[index],filter))
end

@testset "list metadata indexes evaluate ranges" begin
    ivf=IVFIndex(reshape(Float32[1.0,0.0],2,1),[[1,2]])
    index=build_filter_aware_ivf(ivf,[(year=2025,),(year=2026,)])

    @test evaluate_list_filter(index,1,Range(:year,2025,2026))==trues(2)
    @test filtered_list_candidates(index,1,And(Eq(:year,1900),Range(:year,2025,2026)))==Int[]
end
