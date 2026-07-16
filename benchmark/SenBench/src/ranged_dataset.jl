const ARXIV_LARGE_REVISION="79a4cff088965f7039b8766e6115a7769645adb2"
const ARXIV_LARGE_VECTOR_BYTES=44_825_506_432
const ARXIV_LARGE_METADATA_BYTES=1_124_482_479

struct RangedSubsetSpec
    source::String
    revision::String
    vector_etag::String
    vector_count::Int
    dimension::Int
    chunk_records::Int
    seed::Int
end

function ranged_subset_url(
    filename::AbstractString;
    revision::AbstractString = ARXIV_LARGE_REVISION,
)
    return "https://huggingface.co/datasets/SPCL/arxiv-for-fanns-large/resolve/$(revision)/$(filename)?download=true"
end

function disk_available_bytes(path::AbstractString)
    existing=isdir(path) ? String(path) : dirname(abspath(path))
    mkpath(existing)
    lines=split(readchomp(`df -Pk $existing`), '\n')
    length(lines)>=2||throw(ArgumentError("could not read disk capacity"))
    fields=split(last(lines))
    length(fields)>=4||throw(ArgumentError("invalid disk capacity output"))
    return parse(Int, fields[4])*1024
end

function save_ranged_state(path::AbstractString, data::Dict)
    temporary="$(path).tmp"
    open(temporary, "w") do io
        TOML.print(io, data)
    end
    mv(temporary, path; force = true)
    return String(path)
end

function initial_ranged_state(spec::RangedSubsetSpec)
    return Dict{String,Any}(
        "source"=>spec.source,
        "revision"=>spec.revision,
        "vector_etag"=>spec.vector_etag,
        "dimension"=>spec.dimension,
        "completed_records"=>0,
        "chunk_records"=>spec.chunk_records,
        "chunk_starts"=>Int[],
        "chunk_counts"=>Int[],
        "chunk_hashes"=>String[],
    )
end

function load_ranged_state(path::AbstractString, spec::RangedSubsetSpec)
    data=isfile(path) ? TOML.parsefile(path) : initial_ranged_state(spec)
    String(data["source"])==spec.source||throw(ArgumentError("ranged source changed"))
    String(data["revision"])==spec.revision||throw(
        ArgumentError("ranged source revision changed"),
    )
    String(data["vector_etag"])==spec.vector_etag||throw(
        ArgumentError("ranged vector etag changed"),
    )
    Int(data["dimension"])==spec.dimension||throw(
        DimensionMismatch("ranged vector dimension changed"),
    )
    Int(data["chunk_records"])==spec.chunk_records||throw(
        ArgumentError("ranged chunk size changed"),
    )
    return data
end

function download_exact_range(
    url::AbstractString,
    path::AbstractString,
    first_byte::Int,
    last_byte::Int,
)
    first_byte>=0||throw(ArgumentError("first byte cannot be negative"))
    last_byte>=first_byte||throw(ArgumentError("last byte cannot precede first byte"))
    expected=last_byte-first_byte+1
    curl=Sys.which("curl")
    curl===nothing&&throw(ArgumentError("curl is required for ranged downloads"))
    command=Cmd([
        curl,
        "--location",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "4",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "--range",
        "$(first_byte)-$(last_byte)",
        "--max-filesize",
        string(expected),
        "--output",
        String(path),
        String(url),
    ])
    for attempt = 1:6
        isfile(path)&&rm(path)
        process=run(ignorestatus(command))
        success(process)&&isfile(path)&&filesize(path)==expected&&return String(path)
        attempt<6&&println("retrying vector range attempt=$(attempt+1)")
        attempt<6&&sleep(2*attempt)
    end
    throw(ErrorException("ranged download failed after retries"))
end

function append_normalized_fvecs!(
    output_path::AbstractString,
    chunk_path::AbstractString,
    dimension::Int,
    record_count::Int,
)
    record_size=sizeof(Int32)+dimension*sizeof(Float32)
    filesize(chunk_path)==record_count*record_size||throw(
        DimensionMismatch("fvecs chunk size is invalid"),
    )
    vector=Vector{Float32}(undef, dimension)
    open(output_path, "a") do output
        open(chunk_path, "r") do input
            for _ = 1:record_count
                stored_dimension=Int(read(input, Int32))
                stored_dimension==dimension||throw(
                    DimensionMismatch("fvecs chunk dimension changed"),
                )
                read!(input, vector)
                norm=sqrt(sum(abs2, vector))
                iszero(norm)&&throw(ArgumentError("source vector cannot be zero"))
                vector./=norm
                write(output, vector)
            end
            eof(input)||throw(ArgumentError("fvecs chunk contains unexpected bytes"))
        end
        flush(output)
    end
    return String(output_path)
end

function extend_ranged_vectors!(path::AbstractString, spec::RangedSubsetSpec)
    mkpath(path)
    state_path=joinpath(path, "vector_ranges.toml")
    output_path=joinpath(path, "database_vectors.f32")
    temporary_path=joinpath(path, "database_vectors.chunk")
    state=load_ranged_state(state_path, spec)
    completed=Int(state["completed_records"])
    completed<=spec.vector_count||throw(
        ArgumentError("ranged state exceeds requested vector count"),
    )
    expected_output=completed*spec.dimension*sizeof(Float32)

    if isfile(output_path)
        filesize(output_path)>=expected_output||throw(
            ArgumentError("ranged vector output is truncated"),
        )
        filesize(output_path)>expected_output&&truncate(output_path, expected_output)
    else
        iszero(completed)||throw(ArgumentError("ranged vector output is missing"))
        touch(output_path)
    end

    record_size=sizeof(Int32)+spec.dimension*sizeof(Float32)
    source_url=ranged_subset_url("database_vectors.fvecs"; revision = spec.revision)
    while completed<spec.vector_count
        count=min(spec.chunk_records, spec.vector_count-completed)
        first_byte=completed*record_size
        last_byte=(completed+count)*record_size-1
        println("vector range records=$(completed+1)-$(completed+count)")
        download_exact_range(source_url, temporary_path, first_byte, last_byte)
        chunk_hash=file_sha256(temporary_path)
        append_normalized_fvecs!(output_path, temporary_path, spec.dimension, count)
        push!(state["chunk_starts"], completed+1)
        push!(state["chunk_counts"], count)
        push!(state["chunk_hashes"], chunk_hash)
        completed+=count
        state["completed_records"]=completed
        save_ranged_state(state_path, state)
        rm(temporary_path; force = true)
    end
    filesize(output_path)==spec.vector_count*spec.dimension*sizeof(Float32)||throw(
        DimensionMismatch("ranged vector output size is invalid"),
    )
    return (output = output_path, state = state_path, sha256 = file_sha256(output_path))
end

function copy_jsonl_prefix(
    source_path::AbstractString,
    output_path::AbstractString,
    count::Int,
)
    count>0||throw(ArgumentError("jsonl count must be positive"))
    written=0
    temporary="$(output_path).tmp"
    open(temporary, "w") do output
        open(source_path, "r") do input
            for line in eachline(input)
                isempty(strip(line))&&continue
                println(output, line)
                written+=1
                written==count&&break
            end
        end
    end
    written==count||throw(
        ArgumentError("metadata source contains fewer rows than requested"),
    )
    mv(temporary, output_path; force = true)
    return String(output_path)
end

function load_f32_matrix(path::AbstractString, dimension::Int, count::Int)
    dimension>0||throw(ArgumentError("dimension must be positive"))
    count>0||throw(ArgumentError("vector count must be positive"))
    expected=dimension*count*sizeof(Float32)
    isfile(path)&&filesize(path)==expected||throw(
        DimensionMismatch("f32 matrix size is invalid"),
    )
    return open(path, "r") do io
        Mmap.mmap(io, Matrix{Float32}, (dimension, count))
    end
end

function subset_dataset_manifest(path::AbstractString, spec::RangedSubsetSpec)
    filenames=[
        "database_vectors.f32",
        "database_attributes.jsonl",
        "query_vectors.fvecs",
        "em_query_attributes.jsonl",
    ]
    manifest=create_dataset_manifest(
        path;
        name = "arxiv-for-fanns-large-prefix-$(spec.vector_count)",
        source = spec.source,
        revision = spec.revision,
        license = ARXIV_FANNS_LICENSE,
        vector_count = spec.vector_count,
        dimension = spec.dimension,
        query_count = ARXIV_FANNS_QUERY_COUNT,
        metric = :cosine,
        sampling_seed = spec.seed,
        filenames = filenames,
    )
    save_dataset_manifest(joinpath(path, "sen_dataset.toml"), manifest)
    return manifest
end

function prepare_arxiv_subset(
    path::AbstractString;
    vector_count::Int = 300_000,
    chunk_records::Int = 4_096,
    seed::Int = 42,
    revision::AbstractString = ARXIV_LARGE_REVISION,
    keep_metadata_source::Bool = true,
)
    1<=vector_count<=ARXIV_FANNS_COUNTS[:large]||throw(
        ArgumentError("subset vector count is invalid"),
    )
    chunk_records>0||throw(ArgumentError("chunk records must be positive"))
    spec=RangedSubsetSpec(
        "https://huggingface.co/datasets/SPCL/arxiv-for-fanns-large",
        String(revision),
        "72f82ebd2d454483d9000aeb1d507e13a1914037454a7c09c457a043f4b34bf0",
        vector_count,
        ARXIV_FANNS_DIMENSION,
        chunk_records,
        seed,
    )
    mkpath(path)
    existing=isfile(joinpath(path, "database_vectors.f32")) ?
             filesize(joinpath(path, "database_vectors.f32")) : 0
    needed=max(0, vector_count*spec.dimension*sizeof(Float32)-existing)+ARXIV_LARGE_METADATA_BYTES+200_000_000+2*1024^3
    disk_available_bytes(path)>=needed||throw(
        ArgumentError("insufficient disk space for safe subset preparation"),
    )
    vectors=extend_ranged_vectors!(path, spec)
    metadata_source=joinpath(path, "database_attributes.source.jsonl")
    download_file_resumable(
        ranged_subset_url("database_attributes.jsonl"; revision = spec.revision),
        metadata_source,
    )
    copy_jsonl_prefix(
        metadata_source,
        joinpath(path, "database_attributes.jsonl"),
        vector_count,
    )
    keep_metadata_source||rm(metadata_source; force = true)
    download_file_resumable(
        ranged_subset_url("query_vectors.fvecs"; revision = spec.revision),
        joinpath(path, "query_vectors.fvecs"),
    )
    download_file_resumable(
        ranged_subset_url("em_query_attributes.jsonl"; revision = spec.revision),
        joinpath(path, "em_query_attributes.jsonl"),
    )
    manifest=subset_dataset_manifest(path, spec)
    return (spec = spec, manifest = manifest, vectors = vectors)
end

function load_arxiv_subset(
    path::AbstractString,
    train_indices::AbstractVector{<:Integer},
    heldout_indices::AbstractVector{<:Integer};
    k::Int = 10,
    groundtruth = nothing,
    verify_hashes::Bool = false,
    groundtruth_block_size::Int = 2_048,
)
    manifest=load_dataset_manifest(joinpath(path, "sen_dataset.toml"))
    validate_dataset_manifest(manifest, path; verify_hashes = verify_hashes)
    resolved_train=Int.(train_indices)
    resolved_heldout=Int.(heldout_indices)
    all_indices=vcat(resolved_train, resolved_heldout)
    isempty(resolved_train)&&throw(ArgumentError("train query indices cannot be empty"))
    isempty(resolved_heldout)&&throw(ArgumentError("heldout query indices cannot be empty"))
    length(unique(all_indices))==length(all_indices)||throw(
        ArgumentError("query partitions cannot overlap"),
    )
    all(index->1<=index<=manifest.query_count, all_indices)||throw(
        BoundsError(1:manifest.query_count, all_indices),
    )
    vectors=load_f32_matrix(
        joinpath(path, "database_vectors.f32"),
        manifest.dimension,
        manifest.vector_count,
    )
    metadata=load_arxiv_metadata(joinpath(path, "database_attributes.jsonl"))
    length(metadata)==manifest.vector_count||throw(
        DimensionMismatch("metadata count does not match vectors"),
    )
    labels=load_arxiv_query_labels(joinpath(path, "em_query_attributes.jsonl"))
    queries=load_fvecs_indices(joinpath(path, "query_vectors.fvecs"), all_indices)
    filters=NamedTuple[(label = labels[index],) for index in all_indices]
    truths=groundtruth===nothing ?
           load_or_compute_blockwise_groundtruth(
        path,
        manifest,
        vectors,
        metadata,
        queries,
        filters,
        all_indices;
        k = k,
        metric = :cosine,
        block_size = groundtruth_block_size,
    ) : groundtruth
    length(truths)==length(all_indices)||throw(
        DimensionMismatch("groundtruth count does not match queries"),
    )
    train_count=length(resolved_train)
    train_range=1:train_count
    heldout_range=(train_count+1):length(all_indices)
    return ArxivFANNSDataset(
        manifest,
        String(path),
        vectors,
        metadata,
        queries[:, train_range],
        filters[train_range],
        truths[train_range],
        resolved_train,
        queries[:, heldout_range],
        filters[heldout_range],
        truths[heldout_range],
        resolved_heldout,
    )
end
