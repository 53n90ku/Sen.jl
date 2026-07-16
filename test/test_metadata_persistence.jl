using Test
using Sen
using Dates

@testset "metadata persistence" begin
    store=create_metadata_store()

    insert_metadata!(store,(name="first",topic="systems",year=2026,missing_value=missing,))
    insert_metadata!(store,(name="second",active=true,score=0.95,date=Date(2026,7,15),created_at=DateTime(2026,7,15,10,30),tags=["julia","search"],weights=Float32[0.25,0.75],matrix=Int64[1 2;3 4],))

    mktempdir() do path
        metadata_path=save_metadata_store(path,store)

        @test isfile(metadata_path)
        @test read(metadata_path)[1:8]==UInt8[0x53,0x45,0x4e,0x4d,0x45,0x54,0x30,0x32]

        loaded=load_metadata_store(path)

        @test length(loaded)==2
        @test isequal(get_metadata(loaded,1),get_metadata(store,1))
        @test get_metadata(loaded,2)==get_metadata(store,2)
        @test isequal(stored_metadata(loaded),stored_metadata(store))
    end

    empty_store=create_metadata_store()

    mktempdir() do path
        save_metadata_store(path,empty_store)
        loaded=load_metadata_store(path)

        @test length(loaded)==0
        @test isempty(stored_metadata(loaded))
    end

    mktempdir() do path
        @test_throws ArgumentError load_metadata_store(path)
    end

    mktempdir() do path
        metadata_path=joinpath(path,"metadata.bin")

        open(metadata_path,"w") do io
            write(io,UInt8[0x53,0x45,0x4e,0x4d,0x45,0x54,0x30,0x31])
            write(io,Int64(1))
            Sen.Serialization.serialize(io,(name="legacy",year=2025,))
        end

        loaded=load_metadata_store(path)
        @test stored_metadata(loaded)==[(name="legacy",year=2025,)]

        save_metadata_store(path,loaded)
        @test read(metadata_path)[1:8]==UInt8[0x53,0x45,0x4e,0x4d,0x45,0x54,0x30,0x32]
    end

    mktempdir() do path
        unsupported=create_metadata_store()
        insert_metadata!(unsupported,(value=Dict("unsupported"=>true),))
        @test_throws ArgumentError save_metadata_store(path,unsupported)
    end

    mktempdir() do path
        save_metadata_store(path,store)
        metadata_path=joinpath(path,"metadata.bin")

        open(metadata_path,"r+") do io
            truncate(io,12)
        end

        @test_throws ArgumentError load_metadata_store(path)
    end
end
