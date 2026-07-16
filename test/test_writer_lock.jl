using Test
using Sen

@testset "database writer ownership" begin
    mktempdir() do path
        db=create_db(path; dim = 2, metric = :dot)

        @test isopen(db)
        @test isfile(database_writer_lock_path(path))
        @test_throws ArgumentError load_db(path)
        @test_throws ArgumentError create_db(path; dim = 2, metric = :dot)

        close(db)

        @test !isopen(db)
        @test_throws ArgumentError length(db)
        @test_throws ArgumentError insert!(
            db,
            [1.0, 0.0],
            (name = "closed",);
            id = "closed",
        )

        loaded=load_db(path)
        @test isopen(loaded)
        close(loaded)
    end

    mktempdir() do path
        db=create_db(path; dim = 2, metric = :dot, durable = false)
        insert!(db, [1.0, 0.0], (name = "memory",); id = "memory")

        @test !isfile(database_writer_lock_path(path))

        save!(db)

        @test isfile(database_writer_lock_path(path))
        @test_throws ArgumentError load_db(path)

        close(db)
        loaded=load_db(path)
        @test get_record(loaded, "memory").metadata.name=="memory"
        close(loaded)
    end

    mktempdir() do path
        db=create_db(path; dim = 2, metric = :dot)
        project_path=dirname(Base.active_project())
        reject_script="using Sen;try;db=load_db(ARGS[1]);close(db);exit(2);catch error;error isa ArgumentError||rethrow();exit(0);end"
        reject_command=`$(Base.julia_cmd()) --project=$project_path -e $reject_script $path`

        @test success(reject_command)

        close(db)

        accept_script="using Sen;db=load_db(ARGS[1]);close(db)"
        accept_command=`$(Base.julia_cmd()) --project=$project_path -e $accept_script $path`

        @test success(accept_command)
    end
end
