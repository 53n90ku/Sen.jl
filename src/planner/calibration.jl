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

const RECALL_CALIBRATION_MAGIC=UInt8[0x53, 0x45, 0x4e, 0x43, 0x41, 0x4c, 0x30, 0x31]
const RECALL_CALIBRATION_FORMAT_VERSION=1
const RECALL_CALIBRATION_FIELDS=(
    :method,
    :workload,
    :selectivity,
    :vector_count,
    :list_count,
    :target_recall,
    :nprobe,
    :measured_recall,
    :p50_ms,
    :achieved,
    :dimension,
    :metric,
    :index_version,
)

RecallCalibrationEntry(
    method::Symbol,
    workload::Symbol,
    selectivity::Float64,
    vector_count::Int,
    list_count::Int,
    target_recall::Float64,
    nprobe::Int,
    measured_recall::Float64,
    p50_ms::Float64,
    achieved::Bool,
) = RecallCalibrationEntry(
    method,
    workload,
    selectivity,
    vector_count,
    list_count,
    target_recall,
    nprobe,
    measured_recall,
    p50_ms,
    achieved,
    0,
    :cosine,
    1,
)

struct RecallCalibration
    entries::Vector{RecallCalibrationEntry}
end

RecallCalibration() = RecallCalibration(RecallCalibrationEntry[])

function validate_calibration_entry(entry::RecallCalibrationEntry)
    entry.method in (:ivf_prefilter, :ivf_postfilter, :filter_aware, :filter_aware_bound)||throw(
        ArgumentError("unsupported calibration method"),
    )
    entry.workload in (:random, :correlated, :anticorrelated, :skewed, :natural)||throw(
        ArgumentError("unsupported calibration workload"),
    )
    0.0<=entry.selectivity<=1.0||throw(ArgumentError("selectivity must be between 0 and 1"))
    entry.vector_count>0||throw(ArgumentError("vector count must be positive"))
    1<=entry.nprobe<=entry.list_count||throw(
        ArgumentError("nprobe must be between 1 and list count"),
    )
    0.0<entry.target_recall<=1.0||throw(
        ArgumentError("target recall must be between 0 and 1"),
    )
    0.0<=entry.measured_recall<=1.0||throw(
        ArgumentError("measured recall must be between 0 and 1"),
    )
    entry.p50_ms>=0||throw(ArgumentError("p50 latency cannot be negative"))
    entry.dimension>=0||throw(ArgumentError("dimension cannot be negative"))
    entry.metric in (:cosine, :dot)||throw(ArgumentError("unsupported calibration metric"))
    entry.index_version>0||throw(ArgumentError("index version must be positive"))

    return entry
end

function add_calibration_entry!(
    calibration::RecallCalibration,
    entry::RecallCalibrationEntry,
)
    validate_calibration_entry(entry)
    push!(calibration.entries, entry)
    return calibration
end

function calibration_distance(
    entry::RecallCalibrationEntry,
    selectivity::Float64,
    vector_count::Int,
    target_recall::Float64,
)
    selectivity_distance=abs(
        log10(max(entry.selectivity, 1.0e-6))-log10(max(selectivity, 1.0e-6)),
    )
    scale_distance=abs(log10(entry.vector_count)-log10(vector_count))
    recall_distance=abs(entry.target_recall-target_recall)
    return selectivity_distance+0.25*scale_distance+2.0*recall_distance
end

function lookup_calibrated_nprobe(
    calibration::RecallCalibration,
    method::Symbol;
    workload::Symbol = :random,
    selectivity::Float64,
    vector_count::Int,
    list_count::Int,
    dimension::Int = 0,
    metric::Symbol = :cosine,
    index_version::Int = 1,
    target_recall::Float64 = 0.90,
)
    isempty(calibration.entries)&&return nothing
    method in (:ivf_prefilter, :ivf_postfilter, :filter_aware, :filter_aware_bound)||throw(
        ArgumentError("unsupported calibration method"),
    )
    workload in (:random, :correlated, :anticorrelated, :skewed, :natural)||throw(
        ArgumentError("unsupported calibration workload"),
    )
    0.0<=selectivity<=1.0||throw(ArgumentError("selectivity must be between 0 and 1"))
    vector_count>0||throw(ArgumentError("vector count must be positive"))
    list_count>0||throw(ArgumentError("list count must be positive"))
    dimension>=0||throw(ArgumentError("dimension cannot be negative"))
    metric in (:cosine, :dot)||throw(ArgumentError("unsupported calibration metric"))
    index_version>0||throw(ArgumentError("index version must be positive"))
    0.0<target_recall<=1.0||throw(ArgumentError("target recall must be between 0 and 1"))

    candidates=[entry for entry in calibration.entries if entry.method===method]
    isempty(candidates)&&return nothing

    fingerprint_matches=[
        entry for
        entry in candidates if entry.metric===metric&&entry.index_version==index_version&&(
            entry.dimension==0||dimension==0||entry.dimension==dimension
        )
    ]
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

    distances=[
        calibration_distance(entry, selectivity, vector_count, target_recall) for
        entry in candidates
    ]
    entry=candidates[findmin(distances)[2]]
    probe_ratio=entry.nprobe/entry.list_count
    nprobe=clamp(ceil(Int, probe_ratio*list_count), 1, list_count)

    return (
        nprobe = nprobe,
        measured_recall = entry.measured_recall,
        achieved = entry.achieved,
        source = entry,
    )
end

function classify_filter_workload(concentration::Float64)
    0.0<=concentration<=1.0||throw(ArgumentError("concentration must be between 0 and 1"))
    concentration<0.05&&return :random
    concentration<0.25&&return :skewed
    return :correlated
end

function save_recall_calibration(path::AbstractString, calibration::RecallCalibration)
    directory=dirname(path)
    isempty(directory)||mkpath(directory)

    open(path, "w") do io
        write(io, RECALL_CALIBRATION_MAGIC)
        write_portable_uint16(io, RECALL_CALIBRATION_FORMAT_VERSION)
        write_portable_length(io, length(calibration.entries), "calibration entry count")

        for entry in calibration.entries
            validate_calibration_entry(entry)
            write_portable_named_tuple(
                io,
                NamedTuple{RECALL_CALIBRATION_FIELDS}(
                    Tuple(getproperty(entry, field) for field in RECALL_CALIBRATION_FIELDS),
                ),
            )
        end
    end

    return String(path)
end

function calibration_entry_from_portable(value::NamedTuple)
    propertynames(value)==RECALL_CALIBRATION_FIELDS||throw(
        ArgumentError("stored calibration fields are invalid"),
    )
    value.method isa Symbol||throw(ArgumentError("stored calibration method is invalid"))
    value.workload isa Symbol||throw(
        ArgumentError("stored calibration workload is invalid"),
    )
    value.selectivity isa Float64||throw(
        ArgumentError("stored calibration selectivity is invalid"),
    )
    value.vector_count isa Int64||throw(
        ArgumentError("stored calibration vector count is invalid"),
    )
    value.list_count isa Int64||throw(
        ArgumentError("stored calibration list count is invalid"),
    )
    value.target_recall isa Float64||throw(
        ArgumentError("stored calibration target recall is invalid"),
    )
    value.nprobe isa Int64||throw(ArgumentError("stored calibration nprobe is invalid"))
    value.measured_recall isa Float64||throw(
        ArgumentError("stored calibration measured recall is invalid"),
    )
    value.p50_ms isa Float64||throw(ArgumentError("stored calibration latency is invalid"))
    value.achieved isa Bool||throw(
        ArgumentError("stored calibration achieved flag is invalid"),
    )
    value.dimension isa Int64||throw(
        ArgumentError("stored calibration dimension is invalid"),
    )
    value.metric isa Symbol||throw(ArgumentError("stored calibration metric is invalid"))
    value.index_version isa Int64||throw(
        ArgumentError("stored calibration index version is invalid"),
    )

    return validate_calibration_entry(
        RecallCalibrationEntry(
            value.method,
            value.workload,
            value.selectivity,
            Int(value.vector_count),
            Int(value.list_count),
            value.target_recall,
            Int(value.nprobe),
            value.measured_recall,
            value.p50_ms,
            value.achieved,
            Int(value.dimension),
            value.metric,
            Int(value.index_version),
        ),
    )
end

function load_recall_calibration(path::AbstractString)
    isfile(path)||throw(ArgumentError("calibration file does not exist"))

    return open(path, "r") do io
        try
            magic=read(io, length(RECALL_CALIBRATION_MAGIC))

            if magic==RECALL_CALIBRATION_MAGIC
                version=Int(read_portable_uint16(io))
                version==RECALL_CALIBRATION_FORMAT_VERSION||throw(
                    ArgumentError("unsupported calibration format version"),
                )
                count=read_portable_length(io, "calibration entry count")
                entries=Vector{RecallCalibrationEntry}(undef, count)

                for index = 1:count
                    value=read_portable_named_tuple(io)
                    entries[index]=calibration_entry_from_portable(value)
                end

                eof(io)||throw(ArgumentError("calibration file contains unexpected data"))
                return RecallCalibration(entries)
            end

            if is_legacy_julia_serialization_header(magic)
                seekstart(io)
                calibration=Serialization.deserialize(io)
                calibration isa RecallCalibration||throw(
                    ArgumentError("invalid calibration file"),
                )
                return calibration
            end

            throw(ArgumentError("invalid calibration file"))
        catch error
            portable_read_error(error, "calibration file")
        end
    end
end
