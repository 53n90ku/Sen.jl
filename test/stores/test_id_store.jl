using Test
using Sen

@testset "id store" begin
    store=create_id_store(initial_capacity = 1)

    @test length(store)==0
    @test isempty(stored_ids(store))

    first_position=insert_id!(store, "document-1")
    second_position=insert_id!(store, :document_2)
    third_position=insert_id!(store, 300)

    @test first_position==1
    @test second_position==2
    @test third_position==3
    @test length(store)==3

    @test get_id(store, 1)=="document-1"
    @test get_id(store, 2)==:document_2
    @test get_id(store, 3)==300

    @test get_position(store, "document-1")==1
    @test get_position(store, :document_2)==2
    @test get_position(store, 300)==3

    @test stored_ids(store)==["document-1", :document_2, 300]

    @test_throws ArgumentError insert_id!(store, "document-1")
    @test_throws ArgumentError insert_id!(store, nothing)
    @test_throws BoundsError get_id(store, 0)
    @test_throws BoundsError get_id(store, 4)
    @test_throws KeyError get_position(store, "missing")
end

@testset "id persistence" begin
    store=create_id_store()

    insert_id!(store, "document-1")
    insert_id!(store, :document_2)
    insert_id!(store, 300)

    mktempdir() do path
        id_path=save_id_store(path, store)

        @test isfile(id_path)
        @test read(id_path)[1:8]==UInt8[0x53, 0x45, 0x4e, 0x49, 0x44, 0x53, 0x30, 0x32]

        loaded=load_id_store(path)

        @test stored_ids(loaded)==stored_ids(store)
        @test get_position(loaded, "document-1")==1
        @test get_position(loaded, :document_2)==2
        @test get_position(loaded, 300)==3
        @test next_available_id(loaded)==4
    end

    mktempdir() do path
        @test_throws ArgumentError load_id_store(path)
    end

    mktempdir() do path
        id_path=joinpath(path, "ids.bin")

        open(id_path, "w") do io
            write(io, UInt8[0x53, 0x45, 0x4e, 0x49, 0x44, 0x53, 0x30, 0x31])
            write(io, Int64(3))
            Sen.Serialization.serialize(io, "document-1")
            Sen.Serialization.serialize(io, :document_2)
            Sen.Serialization.serialize(io, 300)
        end

        loaded=load_id_store(path)
        @test stored_ids(loaded)==["document-1", :document_2, 300]

        save_id_store(path, loaded)
        @test read(id_path)[1:8]==UInt8[0x53, 0x45, 0x4e, 0x49, 0x44, 0x53, 0x30, 0x32]
    end

    mktempdir() do path
        unsupported=create_id_store()
        insert_id!(unsupported, (value = 1,))
        @test_throws ArgumentError save_id_store(path, unsupported)
    end

    mktempdir() do path
        save_id_store(path, store)
        id_path=joinpath(path, "ids.bin")

        open(id_path, "a") do io
            write(io, UInt8(0xff))
        end

        @test_throws ArgumentError load_id_store(path)
    end
end
