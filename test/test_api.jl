using Test
using Sen

@testset "database insert" begin
    db=create_db("test-db";dim=3,metric=:cosine,initial_capacity=1,durable=false,)

    @test length(db)==0
    @test length(db.vector_store)==0
    @test length(db.metadata_store)==0

    first_id=insert!(db,[1.0,2.0,3.0],(name="first",topic="systems",))
    second_id=insert!(db,Float32[4.0,5.0,6.0],(name="second",topic="databases",))

    @test first_id==1
    @test second_id==2
    @test length(db)==2
    @test length(db.vector_store)==2
    @test length(db.metadata_store)==2

    @test collect(get_vector(db.vector_store,1))==Float32[1.0,2.0,3.0]
    @test get_metadata(db.metadata_store,1)==(name="first",topic="systems",)

    @test_throws DimensionMismatch insert!(db,[1.0,2.0],(name="invalid",))
    @test length(db)==2
    @test length(db.metadata_store)==2
end

@testset "database build" begin
    db=create_db("test-db";dim=2,durable=false,)

    insert!(db,[-1.0,0.0],(side="left",))
    insert!(db,[-0.9,0.1],(side="left",))
    insert!(db,[0.9,0.1],(side="right",))
    insert!(db,[1.0,0.0],(side="right",))

    @test db.index===nothing
    @test db.filter_index===nothing

    build!(db;nlists=2,iterations=20,seed=42,)

    @test db.index isa FilterAwareIVFIndex
    @test db.filter_index isa BitsetIndex
    @test sum(length,db.index.ivf.lists)==4
    @test count(evaluate_filter(db.filter_index,(side="right",)))==2

    insert!(db,[0.8,0.2],(side="right",))

    @test db.index isa FilterAwareIVFIndex
    @test db.filter_index isa BitsetIndex
    @test length(db.delta_store)==1
    @test !is_built(db)
    @test is_dirty(db)

    empty_db=create_db("empty";dim=2,durable=false,)
    @test_throws ArgumentError build!(empty_db;nlists=1,)
end

@testset "database search" begin
    db=create_db("search-db";dim=2,durable=false,)

    insert!(db,[-1.0,0.0],(side="left",name="left-1",))
    insert!(db,[-0.9,0.1],(side="left",name="left-2",))
    insert!(db,[0.9,0.1],(side="right",name="right-1",))
    insert!(db,[1.0,0.0],(side="right",name="right-2",))

    unbuilt_results=search(db,[-1.0,0.0];k=2,nprobe=1,)

    @test length(unbuilt_results)==2
    @test all(result->result isa SearchResult,unbuilt_results)
    @test all(result->result.metadata.side=="left",unbuilt_results)

    build!(db;nlists=2,iterations=20,seed=42,)

    results=search(db,[-1.0,0.0];k=2,nprobe=1,)

    @test length(results)==2
    @test all(result->result isa SearchResult,results)
    @test all(result->result.metadata.side=="left",results)

    filtered_results=search(db,[-1.0,0.0];k=2,nprobe=1,filter=(side="right",),strategy=:filter_aware,vector_weight=0.0,filter_weight=1.0,)

    @test length(filtered_results)==2
    @test all(result->result.metadata.side=="right",filtered_results)
    @test Set(result.index for result in filtered_results)==Set([3,4])

    bound_results=search(db,[-1.0,0.0];k=2,nprobe=1,max_nprobe=2,filter=(side="right",),strategy=:bound,)

    @test Set(result.index for result in bound_results)==Set([3,4])

    prefilter_results=search(db,[-1.0,0.0];k=2,nprobe=2,filter=(side="right",),strategy=:prefilter,)

    @test length(prefilter_results)==2
    @test all(result->result.metadata.side=="right",prefilter_results)

    @test_throws DimensionMismatch search(db,[1.0];k=1,nprobe=1,)

    insert!(db,[0.8,0.2],(side="right",name="right-3",))
    dirty_results=search(db,[1.0,0.0];k=1,nprobe=1,)

    @test only(dirty_results).metadata.name=="right-2"
end

@testset "database ids" begin
    db=create_db("id-db";dim=2,durable=false,)

    first_id=insert!(db,[1.0,0.0],(name="first",);id="document-1",)
    second_id=insert!(db,[0.0,1.0],(name="second",);id=:document_2,)

    @test first_id=="document-1"
    @test second_id==:document_2
    @test length(db.id_store)==2

    @test_throws ArgumentError insert!(db,[-1.0,0.0],(name="duplicate",);id="document-1",)

    @test length(db.vector_store)==2
    @test length(db.metadata_store)==2
    @test length(db.id_store)==2

    build!(db;nlists=2,iterations=20,seed=42,)

    results=search(db,[1.0,0.0];k=2,nprobe=2,strategy=:exact,)

    @test Set(result.id for result in results)==Set(Any["document-1",:document_2])
end

@testset "database batch api" begin
    db=create_db("batch-db";dim=2,durable=false,)
    vectors=Float32[
        1.0 0.0 -1.0
        0.0 1.0 0.0
    ]
    metadata=[
        (name="right",group="a",),
        (name="up",group="b",),
        (name="left",group="a",),
    ]
    ids=insert!(db,vectors,metadata;ids=["right","up","left"],)

    @test ids==["right","up","left"]
    @test length(db)==3
    @test db.revision==1
    @test first(search(db,[1.0,0.0];k=1,)).id=="right"

    updated_vectors=Float32[
        0.0 0.7
        1.0 0.7
    ]
    updated_metadata=[
        (name="right-updated",group="c",),
        (name="diagonal",group="c",),
    ]
    upserted=upsert!(db,updated_vectors,updated_metadata;ids=["right","diagonal"],)

    @test upserted==["right","diagonal"]
    @test length(db)==4
    @test db.revision==2
    @test get_record(db,"right").metadata.name=="right-updated"
    @test get_record(db,"diagonal").vector==Float32[0.7,0.7]

    upsert!(db,[1.0,0.0],(name="right-again",group="a",);id="right",)

    @test get_record(db,"right").metadata.name=="right-again"

    query_results=search(db,Float32[1.0 0.0;0.0 1.0];k=2,)

    @test length(query_results)==2
    @test all(results->all(result->result isa SearchResult,results),query_results)

    delete!(db,["left","up"])

    @test length(db)==2
    @test !has_id(db.id_store,"left")
    @test !has_id(db.id_store,"up")

    state=(count=length(db),revision=db.revision,)

    @test_throws ArgumentError insert!(db,vectors,metadata;ids=["right","new","new"],)
    @test_throws DimensionMismatch insert!(db,vectors,metadata[1:2];ids=[1,2,3],)
    @test_throws KeyError delete!(db,["right","missing"])
    @test length(db)==state.count
    @test db.revision==state.revision
end

@testset "database vector admission" begin
    db=create_db("validation-db";dim=2,metric=:cosine,durable=false,)
    insert!(db,[1.0,0.0],(name="base",);id="base",)
    state=(count=length(db),revision=db.revision,vector=copy(get_record(db,"base").vector),)

    for invalid in ([NaN,1.0],[Inf,1.0],[-Inf,1.0],[floatmax(Float64),1.0],[0.0,-0.0],)
        @test_throws ArgumentError insert!(db,invalid,(name="invalid",))
        @test length(db)==state.count
        @test db.revision==state.revision
    end

    @test_throws DimensionMismatch insert!(db,[1.0],(name="short",))
    @test_throws ArgumentError upsert!(db,[0.0,0.0],(name="invalid",);id="upsert",)
    @test_throws ArgumentError update!(db,"base";vector=[NaN,1.0],)
    @test get_record(db,"base").vector==state.vector
    @test db.revision==state.revision

    invalid_batch=Float64[1.0 NaN;0.0 1.0]
    @test_throws ArgumentError insert!(db,invalid_batch,[(name="valid",),(name="invalid",)];ids=["valid","invalid"],)
    @test length(db)==state.count
    @test db.revision==state.revision
    @test_throws KeyError get_record(db,"valid")

    @test_throws ArgumentError search(db,[NaN,1.0];k=1,)
    @test_throws ArgumentError search(db,[0.0,0.0];k=1,)
    @test_throws ArgumentError search(db,Float64[1.0 Inf;0.0 1.0];k=1,)

    dot_db=create_db("dot-validation-db";dim=2,metric=:dot,durable=false,)
    insert!(dot_db,[0.0,0.0],(name="zero",);id="zero",)
    @test only(search(dot_db,[0.0,0.0];k=1,)).id=="zero"
end
