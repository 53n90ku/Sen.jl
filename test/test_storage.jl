using Test
using Sen

@testset "vector store" begin
    store=create_vector_store(3; initial_capacity = 1)

    @test length(store)==0
    @test size(stored_vectors(store))==(3, 0)

    first_id=insert_vector!(store, [1.0, 2.0, 3.0])
    second_id=insert_vector!(store, Float32[4.0, 5.0, 6.0])

    @test first_id==1
    @test second_id==2
    @test length(store)==2
    @test size(stored_vectors(store))==(3, 2)

    @test collect(get_vector(store, 1))==Float32[1.0, 2.0, 3.0]
    @test collect(get_vector(store, 2))==Float32[4.0, 5.0, 6.0]

    @test_throws DimensionMismatch insert_vector!(store, [1.0, 2.0])
    @test_throws BoundsError get_vector(store, 0)
    @test_throws BoundsError get_vector(store, 3)
end
