using Test
using Sen
using SenBench

using Sen: RecallCalibration, RecallCalibrationEntry, build_bitset_index
using Sen: build_filter_aware_ivf, build_ivf, classify_filter_workload
using Sen: load_recall_calibration, lookup_calibrated_nprobe
using Sen: save_recall_calibration, search_ivf

root=normpath(joinpath(pkgdir(SenBench), "..", ".."))

include(joinpath(@__DIR__, "cases", "test_bench.jl"))
include(joinpath(@__DIR__, "cases", "test_calibration.jl"))
include(joinpath(@__DIR__, "cases", "test_protocol.jl"))
include(joinpath(@__DIR__, "cases", "test_real_data.jl"))
include(joinpath(@__DIR__, "cases", "test_expression_bench.jl"))
