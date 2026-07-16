using Test
using Sen
using Sen: ExactStrategy,PreFilterExactStrategy,logical_filter_stats

@testset "expression planner and dirty statistics" begin
    db=create_db("expression-planner";dim=2,metric=:dot,durable=false,)
    insert!(db,[1.0,0.0],(group="a",score=1,);id="a-1",)
    insert!(db,[0.9,0.0],(group="a",score=2,);id="a-2",)
    insert!(db,[0.8,0.0],(group="b",score=3,);id="b-1",)
    insert!(db,[0.7,0.0],(group="b",score=4,);id="b-2",)
    build!(db;nlists=2,iterations=4,seed=91,)

    insert!(db,[0.95,0.0],(group="z",score=5,);id="z-1",)
    delete!(db,"a-1")

    a_stats=logical_filter_stats(db,Eq(:group,"a"))
    z_stats=logical_filter_stats(db,Eq(:group,"z"))
    not_a_stats=logical_filter_stats(db,Not(Eq(:group,"a")))

    @test a_stats.base_matching_count==1
    @test a_stats.delta_matching_count==0
    @test a_stats.matching_count==1
    @test z_stats.base_matching_count==0
    @test z_stats.delta_matching_count==1
    @test z_stats.matching_count==1
    @test not_a_stats.matching_count==3
    @test not_a_stats.logical_count==4

    exact_plan=plan_query(db,Range(:score,2,5);k=4,strategy=:exact,)
    @test exact_plan.strategy isa PreFilterExactStrategy
    @test exact_plan.estimated_candidates==4
    @test [result.id for result in search(db,[1.0,0.0];k=4,filter=Range(:score,2,5),strategy=:exact,)]==["z-1","a-2","b-1","b-2"]

    queries=Float32[1.0 1.0;0.0 0.0]
    batch=search(db,queries;k=4,filter=Range(:score,2,5),strategy=:exact,parallel=false,)
    @test length(batch)==2
    @test all(results->[result.id for result in results]==["z-1","a-2","b-1","b-2"],batch)
end

@testset "unbuilt strategy validation" begin
    db=create_db("expression-unbuilt";dim=2,metric=:dot,durable=false,)
    insert!(db,[1.0,0.0],(score=1,);id="one",)

    plan=plan_query(db,Range(:score,1,1);strategy=:exact,)
    @test plan.strategy isa PreFilterExactStrategy
    @test_throws ArgumentError search(db,[1.0,0.0];filter=Eq(:score,1),strategy=:prefilter,)
    @test_throws ArgumentError search(db,[1.0,0.0];strategy=:ivf,)
end
