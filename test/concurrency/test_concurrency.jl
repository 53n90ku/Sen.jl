using Test
using Random
using Sen

@testset "database lock" begin
    lock=Sen.DatabaseLock()
    entered=Channel{Nothing}(2)
    release=Channel{Nothing}(2)
    writer_entered=Channel{Nothing}(1)

    readers=[Threads.@spawn Sen.with_database_read(lock) do
        put!(entered, nothing)
        take!(release)
    end for _ = 1:2]

    take!(entered)
    take!(entered)

    writer=Threads.@spawn Sen.with_database_write(lock) do
        put!(writer_entered, nothing)
    end

    yield()

    @test !isready(writer_entered)

    put!(release, nothing)
    put!(release, nothing)
    fetch.(readers)
    take!(writer_entered)
    fetch(writer)

    @test lock.readers==0
    @test lock.writer===nothing
    @test isempty(lock.reader_depth)
end

@testset "concurrent search and mutation" begin
    rng=MersenneTwister(82)
    db=create_db("concurrent-db"; dim = 8, durable = false)

    for id = 1:240
        insert!(db, randn(rng, 8), (group = id%4,); id = id)
    end

    build!(db; nlists = 8, iterations = 4, seed = 42, training_count = 240)
    errors=Channel{Any}(128)
    searchers=[
        Threads.@spawn begin
            local_rng=MersenneTwister(100+task_index)

            for _ = 1:40
                try
                    search(db, randn(local_rng, 8); k = 5, nprobe = 4, strategy = :ivf)
                catch error
                    error isa ArgumentError||put!(errors, error)
                end

                yield()
            end
        end for task_index = 1:6
    ]

    writer=Threads.@spawn begin
        local_rng=MersenneTwister(99)

        for round = 1:10
            update!(db, round; vector = randn(local_rng, 8), metadata = (group = round%4,))
            rebuild!(db)
            yield()
        end
    end

    fetch.(searchers)
    fetch(writer)

    @test !isready(errors)
    @test is_built(db)
    @test db.index_revision==db.revision
    @test length(db)==240
end

@testset "stale index installation" begin
    db=create_db("stale-build"; dim = 2, durable = false)
    insert!(db, [1.0, 0.0], (name = "first",); id = "first")
    insert!(db, [0.0, 1.0], (name = "second",); id = "second")
    build!(db; nlists = 2, iterations = 3, seed = 4)
    stale=(
        revision = db.revision,
        config = db.build_config,
        index = db.index,
        filter_index = db.filter_index,
    )
    update!(db, "first"; vector = [0.5, 0.5])

    @test_throws ArgumentError Sen.install_database_index!(db, stale)
    @test !is_built(db)
    @test db.index!==nothing
    @test length(db.delta_store)==1
end
