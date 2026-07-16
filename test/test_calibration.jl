using Test
using Sen

function calibration_method_result(recall::Float64,p50_ms::Float64)
    return(
        average_recall=recall,
        average_results=10.0,
        average_candidates_visited=100.0,
        average_candidates_scored=10.0,
        average_lists_probed=1.0,
        latency=(minimum_ms=p50_ms,mean_ms=p50_ms,p50_ms=p50_ms,p95_ms=p50_ms,maximum_ms=p50_ms,),
    )
end

function calibration_sweep_result(nprobe::Int,prefilter_recall::Float64,postfilter_recall::Float64,aware_recall::Float64)
    exact=calibration_method_result(1.0,5.0)
    prefilter=calibration_method_result(prefilter_recall,Float64(nprobe))
    postfilter=calibration_method_result(postfilter_recall,Float64(nprobe*2))
    aware=calibration_method_result(aware_recall,Float64(nprobe)+0.5)

    return(
        nprobe=nprobe,
        benchmark=(exact=exact,ivf_prefilter=prefilter,ivf_postfilter=postfilter,filter_aware=aware,average_selectivity=0.05,),
    )
end

@testset "recall calibration" begin
    sweep_results=[
        calibration_sweep_result(1,0.50,0.40,0.60),
        calibration_sweep_result(2,0.91,0.90,0.85),
        calibration_sweep_result(4,1.00,1.00,0.96),
    ]
    calibration=calibrate_recall(sweep_results;workload=:random,selectivity=0.05,vector_count=1000,list_count=4,target_recalls=[0.90],)

    @test length(calibration.entries)==3

    prefilter=lookup_calibrated_nprobe(calibration,:ivf_prefilter;workload=:random,selectivity=0.05,vector_count=1000,list_count=4,target_recall=0.90,)
    aware=lookup_calibrated_nprobe(calibration,:filter_aware;workload=:random,selectivity=0.05,vector_count=2000,list_count=8,target_recall=0.90,)

    @test prefilter.nprobe==2
    @test prefilter.achieved
    @test aware.nprobe==8
    @test aware.measured_recall==0.96

    bound_calibration=calibrate_recall(sweep_results;workload=:random,selectivity=0.05,vector_count=1000,list_count=4,target_recalls=[0.90],methods=[:filter_aware_bound],)
    @test isempty(bound_calibration.entries)

    conservative=calibrate_recall(sweep_results;workload=:random,selectivity=0.05,vector_count=1000,list_count=4,dimension=128,selection_margin=0.03,target_recalls=[0.90],methods=[:ivf_prefilter],)
    @test conservative.entries[1].nprobe==4
    @test lookup_calibrated_nprobe(conservative,:ivf_prefilter;workload=:random,selectivity=0.05,vector_count=1000,list_count=4,dimension=128,target_recall=0.90,)!==nothing
    @test lookup_calibrated_nprobe(conservative,:ivf_prefilter;workload=:random,selectivity=0.05,vector_count=1000,list_count=4,dimension=64,target_recall=0.90,)===nothing

    safety_factor=calibrate_recall(sweep_results;workload=:random,selectivity=0.05,vector_count=1000,list_count=4,probe_safety_factor=2.0,target_recalls=[0.90],methods=[:ivf_prefilter],)
    @test safety_factor.entries[1].nprobe==4

    @test classify_filter_workload(0.01)===:random
    @test classify_filter_workload(0.10)===:skewed
    @test classify_filter_workload(0.50)===:correlated

    mktempdir() do directory
        path=joinpath(directory,"calibration.bin")
        save_recall_calibration(path,calibration)
        @test read(path)[1:8]==UInt8[0x53,0x45,0x4e,0x43,0x41,0x4c,0x30,0x31]
        loaded=load_recall_calibration(path)

        @test loaded.entries==calibration.entries
    end


    mktempdir() do directory
        path=joinpath(directory,"legacy-calibration.bin")

        open(path,"w") do io
            Sen.Serialization.serialize(io,calibration)
        end

        loaded=load_recall_calibration(path)
        @test loaded.entries==calibration.entries

        save_recall_calibration(path,loaded)
        @test read(path)[1:8]==UInt8[0x53,0x45,0x4e,0x43,0x41,0x4c,0x30,0x31]
    end

    mktempdir() do directory
        path=joinpath(directory,"invalid-calibration.bin")
        write(path,"invalid")
        @test_throws ArgumentError load_recall_calibration(path)
    end
end

@testset "heldout calibration evaluation" begin
    vectors=Float32[
        -1.0 -0.9 0.9 1.0
        0.0 0.1 0.1 0.0
    ]
    metadata=[
        (side="left",),
        (side="left",),
        (side="right",),
        (side="right",),
    ]
    queries=reshape(Float32[-1.0,0.0],2,1)
    filters=[(side="right",)]
    context=build_benchmark_context(vectors,metadata;nlists=2,iterations=5,seed=42,)
    calibration=RecallCalibration([
        RecallCalibrationEntry(:ivf_prefilter,:random,0.50,4,2,1.0,2,1.0,1.0,true),
    ])

    evaluation=evaluate_recall_calibration(calibration,context,vectors,metadata,queries,filters,:ivf_prefilter;workload=:random,selectivity=0.50,target_recall=1.0,k=2,repetitions=1,)

    @test evaluation.nprobe==2
    @test evaluation.average_recall==1.0
    @test evaluation.passed
end
