using Test
using Random
using Sen

@testset "database mutations" begin
    db=create_db("mutation-db"; dim = 2, durable = false)
    insert!(db, [1.0, 0.0], (group = "a",); id = "first")
    insert!(db, [0.0, 1.0], (group = "b",); id = "second")
    insert!(db, [-1.0, 0.0], (group = "c",); id = "third")
    build!(db; nlists = 2, iterations = 4, seed = 12, training_count = 3)
    built_revision=db.revision
    config=db.build_config

    @test is_built(db)
    @test !is_dirty(db)

    update!(db, "second"; vector = [0.5, 0.5], metadata = (group = "updated",))

    @test db.revision==built_revision+1
    @test !is_built(db)
    @test is_dirty(db)
    @test db.build_config==config
    @test get_record(db, "second")==(
        id = "second",
        vector = Float32[0.5, 0.5],
        metadata = (group = "updated",),
    )
    @test first(search(db, [0.5, 0.5]; k = 1)).id=="second"

    rebuild!(db)

    @test is_built(db)
    @test db.index_revision==db.revision
    @test first(search(db, [0.5, 0.5]; k = 1, strategy = :exact)).id=="second"

    delete!(db, "first")

    @test length(db)==2
    @test_throws KeyError get_record(db, "first")
    @test db.base_tombstones[get_position(db.id_store, "first")]
    @test get_record(db, "third").vector==Float32[-1.0, 0.0]
    @test !is_built(db)

    state=(
        revision = db.revision,
        vectors = copy(stored_vectors(db.vector_store)),
        metadata = copy(stored_metadata(db.metadata_store)),
        ids = copy(stored_ids(db.id_store)),
    )

    @test_throws KeyError delete!(db, "missing")
    @test_throws KeyError update!(db, "missing"; metadata = (group = "x",))
    @test_throws ArgumentError update!(db, "second")
    @test_throws DimensionMismatch update!(db, "second"; vector = [1.0])
    @test db.revision==state.revision
    @test stored_vectors(db.vector_store)==state.vectors
    @test stored_metadata(db.metadata_store)==state.metadata
    @test stored_ids(db.id_store)==state.ids

    delete!(db, "third")
    delete!(db, "second")

    @test all(db.base_tombstones)
    @test isempty(stored_ids(db.delta_store.id_store))
    @test length(db)==0
end

@testset "mutation state model" begin
    rng=MersenneTwister(91)
    db=create_db("model-db"; dim = 3, durable = false)
    model=Dict{Int,NamedTuple}()
    next_id=1

    for step = 1:160
        operation=isempty(model) ? :insert : rand(rng, (:insert, :insert, :update, :delete))

        if operation===:insert
            id=next_id
            next_id+=1
            vector=Float32.(randn(rng, 3))
            metadata=(group = rand(rng, 1:4), version = step)
            insert!(db, vector, metadata; id = id)
            model[id]=(vector = copy(vector), metadata = metadata)
        elseif operation===:update
            id=rand(rng, collect(keys(model)))
            vector=Float32.(randn(rng, 3))
            metadata=(group = rand(rng, 1:4), version = step)
            update!(db, id; vector = vector, metadata = metadata)
            model[id]=(vector = copy(vector), metadata = metadata)
        else
            id=rand(rng, collect(keys(model)))
            delete!(db, id)
            delete!(model, id)
        end

        @test length(db)==length(model)
        @test length(db.vector_store)==length(db.metadata_store)==length(db.id_store)
        @test length(db.delta_store.vector_store)==length(db.delta_store.metadata_store)==length(
            db.delta_store.id_store,
        )
        live_base=Set(
            id for id in stored_ids(db.id_store) if
            !db.base_tombstones[get_position(db.id_store, id)]
        )
        @test union(live_base, Set(stored_ids(db.delta_store.id_store)))==Set(keys(model))

        for (id, expected) in model
            record=get_record(db, id)
            @test record.id==id
            @test record.vector==expected.vector
            @test record.metadata==expected.metadata

            if has_id(db.delta_store.id_store, id)
                @test get_id(
                    db.delta_store.id_store,
                    get_position(db.delta_store.id_store, id),
                )==id
            else
                @test get_id(db.id_store, get_position(db.id_store, id))==id
            end
        end

        if step%40==0&&!isempty(model)
            nlists=min(4, length(db))
            build!(
                db;
                nlists = nlists,
                iterations = 3,
                seed = step,
                restarts = 1,
                training_count = length(db),
            )
            query=Float32.(randn(rng, 3))
            k=min(5, length(model))
            expected=sort(
                collect(model);
                by = pair->cosine_similarity(query, pair.second.vector),
                rev = true,
            )[1:k]
            actual=search(db, query; k = k, strategy = :exact)
            @test [result.id for result in actual]==[pair.first for pair in expected]
        end
    end

    if !isempty(model)
        nlists=min(4, length(db))
        build!(db; nlists = nlists, iterations = 3, seed = 42, training_count = length(db))
    end

    mktempdir() do path
        db.path=path
        save!(db)
        expected_revision=db.revision
        expected_index_revision=db.index_revision
        expected_built=is_built(db)
        close(db)
        loaded=load_db(path)

        @test loaded.revision==expected_revision
        @test loaded.index_revision==expected_index_revision
        @test is_built(loaded)==expected_built
        @test Set(stored_ids(loaded.id_store))==Set(keys(model))

        for (id, expected) in model
            record=get_record(loaded, id)
            @test record.vector==expected.vector
            @test record.metadata==expected.metadata
        end

        close(loaded)
    end
end

@testset "stale index persistence" begin
    mktempdir() do path
        db=create_db(path; dim = 2)
        insert!(db, [1.0, 0.0], (name = "first",); id = "first")
        insert!(db, [0.0, 1.0], (name = "second",); id = "second")
        build!(db; nlists = 2, iterations = 3, seed = 7)
        update!(db, "first"; vector = [0.5, 0.5])
        save!(db)

        close(db)
        loaded=load_db(path)

        @test !is_built(loaded)
        @test is_dirty(loaded)
        @test loaded.build_config!==nothing
        @test first(search(loaded, [0.5, 0.5]; k = 1)).id=="first"

        rebuild!(loaded)

        @test is_built(loaded)
        @test first(search(loaded, [0.5, 0.5]; k = 1, strategy = :exact)).id=="first"
        close(loaded)
    end
end
