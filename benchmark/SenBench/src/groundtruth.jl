function compute_groundtruth(
    vectors::AbstractMatrix,
    metadata::AbstractVector,
    queries::AbstractMatrix;
    k::Int = 10,
    metric::Symbol = :cosine,
    filters = nothing,
)
    vector_dim, _=size(vectors)
    query_dim, query_count=size(queries)

    vector_dim==query_dim||throw(DimensionMismatch("doesnt match"))
    query_filters = if filters===nothing
        fill(nothing, query_count)
    else
        length(filters)==query_count ||
            throw(DimensionMismatch("filter count dont match query count"))
        filters
    end

    filter_index=build_bitset_index(metadata)
    groundtruth=Vector{Vector{Int}}(undef, query_count)

    for query_index = 1:query_count
        query = @view queries[:, query_index]

        results = search_exact(
            vectors,
            metadata,
            query;
            k = k,
            metric = metric,
            filter = query_filters[query_index],
            filter_index = filter_index,
        )
        groundtruth[query_index]=[result.index for result in results]
    end
    return groundtruth
end

function groundtruth_cache_signature(
    dataset_hash::AbstractString,
    query_indices::AbstractVector{<:Integer},
    k::Int,
    metric::Symbol,
    filter_name::AbstractString,
)
    k>0||throw(ArgumentError("k must be positive"))
    metric in (:cosine, :dot)||throw(ArgumentError("metric must be cosine or dot"))
    value="$(dataset_hash)|$(join(query_indices,','))|$(k)|$(metric)|$(filter_name)"
    return bytes2hex(SHA.sha256(value))
end

function save_ivecs(
    path::AbstractString,
    rows::AbstractVector{<:AbstractVector{<:Integer}};
    zero_based::Bool = true,
)
    isempty(rows)&&throw(ArgumentError("groundtruth rows cannot be empty"))
    dimension=length(first(rows))
    dimension>0||throw(ArgumentError("groundtruth rows cannot be empty"))
    all(row->length(row)==dimension, rows)||throw(
        DimensionMismatch("groundtruth rows must have one dimension"),
    )
    mkpath(dirname(path))

    open(path, "w") do io
        for row in rows
            values=Int32.(row)
            zero_based&&(values .-= 1)
            write(io, Int32(dimension))
            write(io, values)
        end
    end

    return String(path)
end

function save_groundtruth_cache(
    path::AbstractString,
    rows::AbstractVector{<:AbstractVector{<:Integer}};
    dataset_hash::AbstractString,
    query_indices::AbstractVector{<:Integer},
    k::Int,
    metric::Symbol = :cosine,
    filter_name::AbstractString = "none",
)
    length(rows)==length(query_indices)||throw(
        DimensionMismatch("groundtruth rows do not match query indices"),
    )
    data_path="$(path).ivecs"
    manifest_path="$(path).toml"
    signature=groundtruth_cache_signature(
        dataset_hash,
        query_indices,
        k,
        metric,
        filter_name,
    )
    save_ivecs(data_path, rows)
    data=Dict(
        "dataset_hash"=>String(dataset_hash),
        "query_indices"=>Int.(query_indices),
        "k"=>k,
        "metric"=>String(metric),
        "filter_name"=>String(filter_name),
        "signature"=>signature,
        "data_sha256"=>file_sha256(data_path),
    )
    open(manifest_path, "w") do io
        TOML.print(io, data)
    end
    return (data = data_path, manifest = manifest_path, signature = signature)
end

function load_groundtruth_cache(
    path::AbstractString;
    dataset_hash::AbstractString,
    query_indices::AbstractVector{<:Integer},
    k::Int,
    metric::Symbol = :cosine,
    filter_name::AbstractString = "none",
)
    data_path="$(path).ivecs"
    manifest_path="$(path).toml"
    isfile(data_path)&&isfile(manifest_path)||throw(
        ArgumentError("groundtruth cache does not exist"),
    )
    data=TOML.parsefile(manifest_path)
    expected=groundtruth_cache_signature(
        dataset_hash,
        query_indices,
        k,
        metric,
        filter_name,
    )
    String(data["signature"])==expected||throw(ArgumentError("groundtruth cache is stale"))
    file_sha256(data_path)==String(data["data_sha256"])||throw(
        ArgumentError("groundtruth cache data changed"),
    )
    return load_ivecs_indices(data_path, collect(1:length(query_indices)); k = k)
end
