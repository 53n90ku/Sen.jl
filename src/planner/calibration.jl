struct RecallCalibrationEntry
    method::Symbol
    workload::Symbol
    selectivity::Float64
    vector_count::Int
    list_count::Int
    target_recall::Float64
    nprobe::Int
    measured_recall::Float64
    p50_ms::Float64
    achieved::Bool
    dimension::Int
    metric::Symbol
    index_version::Int
end

RecallCalibrationEntry(method::Symbol,workload::Symbol,selectivity::Float64,vector_count::Int,list_count::Int,target_recall::Float64,nprobe::Int,measured_recall::Float64,p50_ms::Float64,achieved::Bool)=RecallCalibrationEntry(method,workload,selectivity,vector_count,list_count,target_recall,nprobe,measured_recall,p50_ms,achieved,0,:cosine,1)

struct RecallCalibration
    entries::Vector{RecallCalibrationEntry}
end

RecallCalibration()=RecallCalibration(RecallCalibrationEntry[])

function validate_calibration_entry(entry::RecallCalibrationEntry)
    entry.method in (:ivf_prefilter,:ivf_postfilter,:filter_aware,:filter_aware_bound)||throw(ArgumentError("unsupported calibration method"))
    entry.workload in (:random,:correlated,:anticorrelated,:skewed,:natural)||throw(ArgumentError("unsupported calibration workload"))
    0.0<=entry.selectivity<=1.0||throw(ArgumentError("selectivity must be between 0 and 1"))
    entry.vector_count>0||throw(ArgumentError("vector count must be positive"))
    1<=entry.nprobe<=entry.list_count||throw(ArgumentError("nprobe must be between 1 and list count"))
    0.0<entry.target_recall<=1.0||throw(ArgumentError("target recall must be between 0 and 1"))
    0.0<=entry.measured_recall<=1.0||throw(ArgumentError("measured recall must be between 0 and 1"))
    entry.p50_ms>=0||throw(ArgumentError("p50 latency cannot be negative"))
    entry.dimension>=0||throw(ArgumentError("dimension cannot be negative"))
    entry.metric in (:cosine,:dot)||throw(ArgumentError("unsupported calibration metric"))
    entry.index_version>0||throw(ArgumentError("index version must be positive"))

    return entry
end

function add_calibration_entry!(calibration::RecallCalibration,entry::RecallCalibrationEntry)
    validate_calibration_entry(entry)
    push!(calibration.entries,entry)
    return calibration
end

function calibration_distance(entry::RecallCalibrationEntry,selectivity::Float64,vector_count::Int,target_recall::Float64)
    selectivity_distance=abs(log10(max(entry.selectivity,1.0e-6))-log10(max(selectivity,1.0e-6)))
    scale_distance=abs(log10(entry.vector_count)-log10(vector_count))
    recall_distance=abs(entry.target_recall-target_recall)
    return selectivity_distance+0.25*scale_distance+2.0*recall_distance
end

function lookup_calibrated_nprobe(calibration::RecallCalibration,method::Symbol;workload::Symbol=:random,selectivity::Float64,vector_count::Int,list_count::Int,dimension::Int=0,metric::Symbol=:cosine,index_version::Int=1,target_recall::Float64=0.90,)
    isempty(calibration.entries)&&return nothing
    method in (:ivf_prefilter,:ivf_postfilter,:filter_aware,:filter_aware_bound)||throw(ArgumentError("unsupported calibration method"))
    workload in (:random,:correlated,:anticorrelated,:skewed,:natural)||throw(ArgumentError("unsupported calibration workload"))
    0.0<=selectivity<=1.0||throw(ArgumentError("selectivity must be between 0 and 1"))
    vector_count>0||throw(ArgumentError("vector count must be positive"))
    list_count>0||throw(ArgumentError("list count must be positive"))
    dimension>=0||throw(ArgumentError("dimension cannot be negative"))
    metric in (:cosine,:dot)||throw(ArgumentError("unsupported calibration metric"))
    index_version>0||throw(ArgumentError("index version must be positive"))
    0.0<target_recall<=1.0||throw(ArgumentError("target recall must be between 0 and 1"))

    candidates=[entry for entry in calibration.entries if entry.method===method]
    isempty(candidates)&&return nothing

    fingerprint_matches=[entry for entry in candidates if entry.metric===metric&&entry.index_version==index_version&&(entry.dimension==0||dimension==0||entry.dimension==dimension)]
    isempty(fingerprint_matches)&&return nothing
    candidates=fingerprint_matches

    workload_matches=[entry for entry in candidates if entry.workload===workload]
    if !isempty(workload_matches)
        candidates=workload_matches
    end

    recall_matches=[entry for entry in candidates if entry.target_recall>=target_recall]
    if !isempty(recall_matches)
        candidates=recall_matches
    end

    achieved_matches=[entry for entry in candidates if entry.achieved]
    if !isempty(achieved_matches)
        candidates=achieved_matches
    end

    distances=[calibration_distance(entry,selectivity,vector_count,target_recall) for entry in candidates]
    entry=candidates[findmin(distances)[2]]
    probe_ratio=entry.nprobe/entry.list_count
    nprobe=clamp(ceil(Int,probe_ratio*list_count),1,list_count)

    return(
        nprobe=nprobe,
        measured_recall=entry.measured_recall,
        achieved=entry.achieved,
        source=entry,
    )
end

function classify_filter_workload(concentration::Float64)
    0.0<=concentration<=1.0||throw(ArgumentError("concentration must be between 0 and 1"))
    concentration<0.05&&return :random
    concentration<0.25&&return :skewed
    return :correlated
end

function save_recall_calibration(path::AbstractString,calibration::RecallCalibration)
    directory=dirname(path)
    isempty(directory)||mkpath(directory)

    open(path,"w") do io
        serialize(io,calibration)
    end

    return String(path)
end

function load_recall_calibration(path::AbstractString)
    isfile(path)||throw(ArgumentError("calibration file does not exist"))

    calibration=open(path,"r") do io
        deserialize(io)
    end
    calibration isa RecallCalibration||throw(ArgumentError("invalid calibration file"))

    return calibration
end
