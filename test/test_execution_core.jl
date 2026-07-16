using Test
using Random
using Sen

@testset "bounded top k oracle" begin
    rng=MersenneTwister(801)

    for count in (1,2,5,16,65)
        for k in 1:count
            scores=Float32.(rand(rng,0:7,count))
            expected=collect(partialsortperm(scores,1:k;rev=true,))
            actual=[result.index for result in top_k(scores,k)]
            @test actual==expected
        end
    end

    special=Float32[NaN,1.0,NaN,-0.0,0.0]

    for k in 1:length(special)
        expected=collect(partialsortperm(special,1:k;rev=true,))
        actual=[result.index for result in top_k(special,k)]
        @test actual==expected
    end
end

@testset "streaming score oracle" begin
    rng=MersenneTwister(802)
    vectors=randn(rng,Float32,32,300)
    metadata=[(number=index,) for index in 1:300]
    query=randn(rng,Float32,32)
    norms=compute_vector_norms(vectors)
    candidates=shuffle(rng,collect(1:300))
    excluded=falses(300)
    excluded[randperm(rng,300)[1:40]].=true
    retained=Int[index for index in candidates if !excluded[index]]

    for metric in (:dot,:cosine)
        scores=Float32[]
        query_norm=vector_norm(query)

        for index in retained
            score=column_dot(query,vectors,index)
            metric===:cosine&&(score/=query_norm*norms[index])
            push!(scores,score)
        end

        order=partialsortperm(scores,1:25;rev=true,)
        expected_indices=retained[order]
        expected_scores=scores[order]
        actual=score_ivf_candidates(vectors,metadata,query,candidates;k=25,metric=metric,vector_norms=norms,excluded=excluded,)

        @test [result.index for result in actual]==expected_indices
        @test [result.score for result in actual]==expected_scores
    end
end

@testset "blocked exact score oracle" begin
    rng=MersenneTwister(803)
    vectors=randn(rng,Float32,64,1_024)
    metadata=[(number=index,) for index in 1:1_024]
    query=randn(rng,Float32,64)
    norms=compute_vector_norms(vectors)
    scalar_scores=Float32[column_dot(query,vectors,index)/(vector_norm(query)*norms[index]) for index in 1:1_024]
    expected=collect(partialsortperm(scalar_scores,1:20;rev=true,))
    actual=search_exact(vectors,metadata,query;k=20,metric=:cosine,vector_norms=norms,)

    @test [result.index for result in actual]==expected
    @test [result.score for result in actual]≈scalar_scores[expected]
end

@testset "search allocation ceilings" begin
    rng=MersenneTwister(804)
    count=2_000
    dim=32
    vectors=randn(rng,Float32,dim,count)
    metadata=[(group=index%5,) for index in 1:count]
    query=randn(rng,Float32,dim)
    queries=randn(rng,Float32,dim,8)
    db=create_db("allocation-core";dim=dim,durable=false,maintenance_config=MaintenanceConfig(enabled=false,),)

    insert!(db,vectors,metadata;ids=collect(1:count),)
    build!(db;nlists=32,iterations=3,seed=804,)
    search(db,query;k=10,strategy=:exact,)
    search(db,query;k=10,nprobe=8,strategy=:ivf,)
    search(db,queries;k=10,nprobe=8,strategy=:ivf,parallel=false,)

    exact_allocated=@allocated search(db,query;k=10,strategy=:exact,)
    ivf_allocated=@allocated search(db,query;k=10,nprobe=8,strategy=:ivf,)
    batch_allocated=@allocated search(db,queries;k=10,nprobe=8,strategy=:ivf,parallel=false,)

    @test exact_allocated<50_000
    @test ivf_allocated<35_000
    @test batch_allocated<250_000
    close(db)
end

@testset "blocked matrix exact search" begin
    rng=MersenneTwister(805)
    vectors=randn(rng,Float32,48,1_200)
    metadata=[(group=index%7,) for index in 1:1_200]
    queries=randn(rng,Float32,48,12)
    db=create_db("matrix-exact";dim=48,durable=false,maintenance_config=MaintenanceConfig(enabled=false,),)

    insert!(db,vectors,metadata;ids=collect(1:1_200),)
    build!(db;nlists=24,iterations=3,seed=805,)
    matrix_results=search(db,queries;k=12,strategy=:exact,parallel=false,)
    scalar_results=[search(db,@view(queries[:,index]);k=12,strategy=:exact,) for index in axes(queries,2)]

    @test [[result.id for result in results] for results in matrix_results]==[[result.id for result in results] for results in scalar_results]
    @test all(index->[result.score for result in matrix_results[index]]≈[result.score for result in scalar_results[index]],axes(queries,2))

    insert!(db,randn(rng,Float32,48),(group=1,);id=1_201,)
    dirty_matrix=search(db,queries;k=12,strategy=:exact,parallel=true,workers=3,)
    dirty_scalar=[search(db,@view(queries[:,index]);k=12,strategy=:exact,) for index in axes(queries,2)]

    @test [[result.id for result in results] for results in dirty_matrix]==[[result.id for result in results] for results in dirty_scalar]
    @test all(index->[result.score for result in dirty_matrix[index]]≈[result.score for result in dirty_scalar[index]],axes(queries,2))

    search(db,queries;k=12,strategy=:exact,parallel=false,)
    allocated=@allocated search(db,queries;k=12,strategy=:exact,parallel=false,)

    @test allocated<100_000
    close(db)
end
