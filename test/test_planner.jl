using Test
using Sen

@testset "planner cost model" begin
    config=PlannerConfig()

    @test config.target_recall==0.90
    @test config.candidate_multiplier==4.0
    @test config.postfilter_candidate_multiplier==4.0
    @test config.default_nprobe==4
    @test config.max_nprobe==64
    @test config.exact_filter_threshold==0.01
    @test config.calibration===nothing
    @test config.workload===:auto

    @test estimate_candidate_count(1000, 0.0)==0
    @test estimate_candidate_count(1000, 0.004)==4
    @test estimate_candidate_count(1000, 0.10)==100
    @test estimate_required_nprobe(
        1000,
        10,
        0.10,
        10;
        candidate_multiplier = 1.0,
        target_recall = 1.0,
    )==1
    @test estimate_required_nprobe(
        1000,
        10,
        0.01,
        10;
        candidate_multiplier = 1.0,
        target_recall = 1.0,
    )==10

    @test_throws ArgumentError PlannerConfig(target_recall = 0.0)
    @test_throws ArgumentError PlannerConfig(default_nprobe = 8, max_nprobe = 4)
    @test_throws ArgumentError PlannerConfig(workload = :missing)
    @test_throws ArgumentError PlannerConfig(exact_filter_threshold = 1.1)
    @test_throws ArgumentError estimate_candidate_count(-1, 0.5)
    @test_throws ArgumentError estimate_candidate_count(100, 1.5)
end

@testset "planner cache" begin
    db=create_db("planner-cache"; dim = 4, initial_capacity = 40, durable = false)

    for index = 1:40
        insert!(db, Float32[index, 1.0, 0.0, 0.0], (selected = index<=10,))
    end

    build!(db; nlists = 4, iterations = 3, seed = 42)
    config=PlannerConfig(target_recall = 0.90, max_nprobe = 4)
    query=Float32[1.0, 0.0, 0.0, 0.0]

    @test isempty(db.plan_cache)
    search(db, query; k = 3, filter = (selected = true,), planner_config = config)
    @test length(db.plan_cache)==1
    search(db, query; k = 3, filter = (selected = true,), planner_config = config)
    @test length(db.plan_cache)==1

    update!(db, 1; metadata = (selected = false,))
    @test isempty(db.plan_cache)
end

@testset "strategy selection" begin
    db=create_db("planner-db"; dim = 2, initial_capacity = 100, durable = false)

    for index = 1:100
        insert!(
            db,
            Float32[index, 1.0],
            (group = index<=10 ? "rare" : "common", unique = index==1),
        )
    end

    build!(db; nlists = 4, iterations = 5, seed = 42)

    no_filter_plan=choose_strategy(db, nothing; k = 5)
    filtered_plan=choose_strategy(db, (group = "rare",); k = 5)
    unique_plan=choose_strategy(db, (unique = true,); k = 1)
    estimates=estimate_strategy_costs(db, (group = "rare",); k = 5)

    @test no_filter_plan.strategy isa IVFStrategy
    @test filtered_plan.strategy isa SearchStrategy
    @test unique_plan.strategy isa PreFilterExactStrategy
    @test unique_plan.selectivity==0.01
    @test filtered_plan.selectivity==0.10
    @test filtered_plan.estimated_candidates==10
    @test 0<=filtered_plan.nprobe<=4
    @test 0<=filtered_plan.minimum_nprobe<=filtered_plan.nprobe
    @test filtered_plan.estimated_cost>=0
    @test estimates.required_nprobe>=1
    @test estimates.required_nprobe>=estimates.uniform_nprobe
    @test 0.0<=estimates.concentration<=1.0

    manual_exact=plan_query(db, (group = "rare",); k = 5, strategy = :exact)
    manual_prefilter=plan_query(db, (group = "rare",); k = 5, strategy = :prefilter)
    manual_postfilter=plan_query(db, (group = "rare",); k = 5, strategy = :postfilter)
    manual_filter_aware=plan_query(db, (group = "rare",); k = 5, strategy = :filter_aware)
    manual_bound=plan_query(db, (group = "rare",); k = 5, strategy = :bound)

    @test manual_exact.strategy isa PreFilterExactStrategy
    @test manual_prefilter.strategy isa IVFPreFilterStrategy
    @test manual_postfilter.strategy isa IVFPostFilterStrategy
    @test manual_filter_aware.strategy isa FilterAwareIVFStrategy
    @test manual_bound.strategy isa BoundedFilterAwareIVFStrategy

    calibration=RecallCalibration([
        RecallCalibrationEntry(
            :filter_aware,
            :random,
            0.10,
            100,
            4,
            0.90,
            4,
            0.95,
            1.0,
            true,
        ),
        RecallCalibrationEntry(
            :filter_aware_bound,
            :random,
            0.10,
            100,
            4,
            0.90,
            2,
            0.95,
            0.8,
            true,
        ),
    ])
    calibrated_config=PlannerConfig(calibration = calibration, workload = :random)
    calibrated_estimates=estimate_strategy_costs(
        db,
        (group = "rare",);
        k = 5,
        config = calibrated_config,
    )
    calibrated_plan=plan_query(
        db,
        (group = "rare",);
        k = 5,
        strategy = :filter_aware,
        config = calibrated_config,
    )

    @test calibrated_estimates.filter_aware_calibrated
    @test calibrated_estimates.filter_aware_bound_calibrated
    @test calibrated_estimates.filter_aware_bound_max_nprobe==2
    @test calibrated_estimates.filter_aware_min_nprobe==4
    @test calibrated_plan.minimum_nprobe==4
    @test calibrated_plan.nprobe==4

    @test_throws ArgumentError plan_query(db, (group = "rare",); strategy = :ivf)
    @test_throws ArgumentError plan_query(db, nothing; strategy = :postfilter)
    @test_throws ArgumentError plan_query(db, nothing; strategy = :unknown)

    unbuilt_db=create_db("unbuilt"; dim = 2, durable = false)
    unbuilt_plan=choose_strategy(unbuilt_db, (topic = "systems",))
    @test unbuilt_plan.strategy isa PreFilterExactStrategy
end
