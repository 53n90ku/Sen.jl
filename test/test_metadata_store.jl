using Test
using Sen

@testset "metadata store" begin
    store=create_metadata_store(initial_capacity=1,)

    @test length(store)==0
    @test isempty(stored_metadata(store))

    first_id=insert_metadata!(store,(name="first",topic="systems",))
    second_id=insert_metadata!(store,(name="second",year=2026,))

    @test first_id==1
    @test second_id==2
    @test length(store)==2

    @test get_metadata(store,1)==(name="first",topic="systems",)
    @test get_metadata(store,2)==(name="second",year=2026,)
    @test stored_metadata(store)==[
        (name="first",topic="systems",),
        (name="second",year=2026,),
    ]

    @test_throws BoundsError get_metadata(store,0)
    @test_throws BoundsError get_metadata(store,3)
    @test_throws ArgumentError create_metadata_store(initial_capacity=-1,)
end