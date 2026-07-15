const ARXIV_FANNS_FILES=[
    "database_vectors.fvecs",
    "query_vectors.fvecs",
    "database_attributes.jsonl",
    "em_query_attributes.jsonl",
    "ground_truth_em.ivecs",
]

const ARXIV_FANNS_COUNTS=Dict(
    :small=>1_000,
    :medium=>100_000,
    :large=>2_735_264,
)

const ARXIV_FANNS_QUERY_COUNT=10_000
const ARXIV_FANNS_DIMENSION=4_096
const ARXIV_FANNS_LICENSE="dataset card declares no blanket license; source records retain mixed arXiv licenses"

struct RealQuerySplit
    train_indices::Vector{Int}
    heldout_indices::Vector{Int}
end

Base.:(==)(left::RealQuerySplit,right::RealQuerySplit)=left.train_indices==right.train_indices&&left.heldout_indices==right.heldout_indices

struct RealQueryPartitions
    development_indices::Vector{Int}
    validation_indices::Vector{Int}
    confirmation_indices::Vector{Int}
    confirmation_count::Int
    confirmation_hash::String
    seed::Int
end

Base.:(==)(left::RealQueryPartitions,right::RealQueryPartitions)=left.development_indices==right.development_indices&&left.validation_indices==right.validation_indices&&left.confirmation_indices==right.confirmation_indices&&left.confirmation_count==right.confirmation_count&&left.confirmation_hash==right.confirmation_hash&&left.seed==right.seed

struct ArxivFANNSDataset
    manifest::DatasetManifest
    path::String
    vectors::Matrix{Float32}
    metadata::Vector{NamedTuple}
    train_queries::Matrix{Float32}
    train_filters::Vector{NamedTuple}
    train_truth::Vector{Vector{Int}}
    train_indices::Vector{Int}
    heldout_queries::Matrix{Float32}
    heldout_filters::Vector{NamedTuple}
    heldout_truth::Vector{Vector{Int}}
    heldout_indices::Vector{Int}
end

function arxiv_fanns_repository(scale::Symbol)
    haskey(ARXIV_FANNS_COUNTS,scale)||throw(ArgumentError("scale must be small, medium or large"))
    return "SPCL/arxiv-for-fanns-$(scale)"
end

function arxiv_fanns_url(scale::Symbol,filename::AbstractString)
    filename in ARXIV_FANNS_FILES||throw(ArgumentError("unknown arxiv-for-fanns file"))
    return "https://huggingface.co/datasets/$(arxiv_fanns_repository(scale))/resolve/main/$(filename)?download=true"
end

function download_file_resumable(url::AbstractString,path::AbstractString)
    isfile(path)&&filesize(path)>0&&return String(path)
    curl=Sys.which("curl")
    curl===nothing&&throw(ArgumentError("curl is required for resumable dataset downloads"))
    mkpath(dirname(path))
    partial="$(path).part"
    command=Cmd([curl,"--location","--fail","--retry","4","--retry-delay","2","--continue-at","-","--output",partial,String(url)])
    run(command)
    filesize(partial)>0||throw(ArgumentError("downloaded file is empty"))
    mv(partial,path;force=true,)
    return String(path)
end

function download_arxiv_fanns(path::AbstractString;scale::Symbol=:medium,)
    arxiv_fanns_repository(scale)
    mkpath(path)
    downloaded=Dict{String,String}()

    for filename in ARXIV_FANNS_FILES
        println("dataset file=$(filename)")
        downloaded[filename]=download_file_resumable(arxiv_fanns_url(scale,filename),joinpath(path,filename))
    end

    return downloaded
end

function vector_file_info(path::AbstractString,stored_type::Type)
    isfile(path)||throw(ArgumentError("vector file does not exist"))
    filesize(path)>sizeof(Int32)||throw(ArgumentError("vector file is empty"))
    return open(path,"r") do io
        dimension=Int(read(io,Int32))
        dimension>0||throw(ArgumentError("stored dimension must be positive"))
        record_size=sizeof(Int32)+dimension*sizeof(stored_type)
        filesize(path)%record_size==0||throw(ArgumentError("vector file size does not match its dimension"))
        return(dimension=dimension,count=filesize(path)÷record_size,record_size=record_size,)
    end
end

fvecs_info(path::AbstractString)=vector_file_info(path,Float32)

function ivecs_info(path::AbstractString)
    isfile(path)||throw(ArgumentError("ivecs file does not exist"))
    count=0
    minimum_dimension=typemax(Int)
    maximum_dimension=0

    open(path,"r") do io
        while !eof(io)
            bytesavailable=filesize(path)-position(io)
            bytesavailable>=sizeof(Int32)||throw(ArgumentError("ivecs record is truncated"))
            dimension=Int(read(io,Int32))
            dimension>=0||throw(ArgumentError("ivecs dimension cannot be negative"))
            remaining=filesize(path)-position(io)
            dimension*sizeof(Int32)<=remaining||throw(ArgumentError("ivecs record is truncated"))
            skip(io,dimension*sizeof(Int32))
            count+=1
            minimum_dimension=min(minimum_dimension,dimension)
            maximum_dimension=max(maximum_dimension,dimension)
        end
    end

    count>0||throw(ArgumentError("ivecs file is empty"))
    return(dimension=maximum_dimension,minimum_dimension=minimum_dimension,count=count,variable=minimum_dimension!=maximum_dimension,)
end

function load_fvecs_indices(path::AbstractString,indices::AbstractVector{<:Integer};normalize::Bool=true,)
    info=fvecs_info(path)
    resolved=Int.(indices)
    all(index->1<=index<=info.count,resolved)||throw(BoundsError(1:info.count,resolved))
    vectors=Matrix{Float32}(undef,info.dimension,length(resolved))

    open(path,"r") do io
        for(output_index,source_index) in enumerate(resolved)
            seek(io,(source_index-1)*info.record_size)
            stored_dimension=Int(read(io,Int32))
            stored_dimension==info.dimension||throw(DimensionMismatch("fvecs vectors must have one dimension"))
            read!(io,@view vectors[:,output_index])
        end
    end

    normalize&&normalize_columns!(vectors)
    return vectors
end

function load_ivecs_indices(path::AbstractString,indices::AbstractVector{<:Integer};k::Union{Nothing,Int}=nothing,zero_based::Bool=true,)
    info=ivecs_info(path)
    resolved=Int.(indices)
    all(index->1<=index<=info.count,resolved)||throw(BoundsError(1:info.count,resolved))
    k===nothing||k>0||throw(ArgumentError("k must be positive"))
    rows=Vector{Vector{Int}}(undef,length(resolved))
    outputs=Dict{Int,Vector{Int}}()
    for(output_index,source_index) in enumerate(resolved)
        push!(get!(outputs,source_index,Int[]),output_index)
    end

    open(path,"r") do io
        source_index=0
        while !eof(io)&&!isempty(outputs)
            source_index+=1
            dimension=Int(read(io,Int32))

            if haskey(outputs,source_index)
                values=Vector{Int32}(undef,dimension)
                read!(io,values)
                row=Int[value for value in values if value>=0]
                zero_based&&(row .+=1)
                selected=k===nothing ? row : row[1:min(k,length(row))]
                for output_index in outputs[source_index]
                    rows[output_index]=copy(selected)
                end
                delete!(outputs,source_index)
            else
                skip(io,dimension*sizeof(Int32))
            end
        end
    end

    isempty(outputs)||throw(ArgumentError("ivecs indices could not be loaded"))
    return rows
end

function load_arxiv_metadata(path::AbstractString;limit::Union{Nothing,Int}=nothing,)
    isfile(path)||throw(ArgumentError("metadata file does not exist"))
    limit===nothing||limit>0||throw(ArgumentError("limit must be positive"))
    metadata=NamedTuple[]

    open(path,"r") do io
        for line in eachline(io)
            isempty(strip(line))&&continue
            row=JSON3.read(line)
            categories=String.(row.main_categories)
            isempty(categories)&&throw(ArgumentError("paper must contain a main category"))
            push!(metadata,(
                label=Int(row.number_of_sub_categories),
                main_category=first(categories),
                has_comments=Bool(row.has_comments),
                version_count=Int(row.number_of_versions),
                author_count=Int(row.number_of_authors),
                license=row.license===nothing ? "unspecified" : String(row.license),
            ))
            limit!==nothing&&length(metadata)==limit&&break
        end
    end

    isempty(metadata)&&throw(ArgumentError("metadata file is empty"))
    limit===nothing||length(metadata)==limit||throw(ArgumentError("metadata contains fewer rows than requested"))
    return metadata
end

function load_arxiv_query_labels(path::AbstractString)
    isfile(path)||throw(ArgumentError("query attribute file does not exist"))
    labels=Int[]

    open(path,"r") do io
        for line in eachline(io)
            isempty(strip(line))&&continue
            row=JSON3.read(line)
            push!(labels,Int(row.label))
        end
    end

    isempty(labels)&&throw(ArgumentError("query attribute file is empty"))
    return labels
end

function stratified_query_split(labels::AbstractVector{<:Integer};train_count::Int,heldout_count::Int,seed::Int=42,)
    train_count>0||throw(ArgumentError("train count must be positive"))
    heldout_count>0||throw(ArgumentError("heldout count must be positive"))
    train_count+heldout_count<=length(labels)||throw(ArgumentError("not enough queries for split"))
    rng=MersenneTwister(seed)
    groups=Dict{Int,Vector{Int}}()

    for(index,label) in enumerate(labels)
        push!(get!(groups,Int(label),Int[]),index)
    end

    for group in values(groups)
        shuffle!(rng,group)
    end

    function take_stratified!(count::Int)
        selected=Int[]
        ordered_labels=sort!(collect(keys(groups)))

        while length(selected)<count
            added=false
            for label in ordered_labels
                isempty(groups[label])&&continue
                push!(selected,pop!(groups[label]))
                added=true
                length(selected)==count&&break
            end
            added||throw(ArgumentError("not enough queries for stratified split"))
        end

        shuffle!(rng,selected)
        return selected
    end

    train_indices=take_stratified!(train_count)
    heldout_indices=take_stratified!(heldout_count)
    return RealQuerySplit(train_indices,heldout_indices)
end

function stratified_query_partitions(labels::AbstractVector{<:Integer};development_count::Int=256,validation_count::Int=128,confirmation_count::Int=256,seed::Int=42,strata::AbstractVector{<:Integer}=labels,)
    development_count>0||throw(ArgumentError("development count must be positive"))
    validation_count>0||throw(ArgumentError("validation count must be positive"))
    confirmation_count>0||throw(ArgumentError("confirmation count must be positive"))
    total=development_count+validation_count+confirmation_count
    total<=length(labels)||throw(ArgumentError("not enough queries for partitions"))
    length(strata)==length(labels)||throw(DimensionMismatch("query strata do not match labels"))
    first_split=stratified_query_split(strata;train_count=development_count,heldout_count=validation_count+confirmation_count,seed=seed,)
    heldout_strata=strata[first_split.heldout_indices]
    second_split=stratified_query_split(heldout_strata;train_count=validation_count,heldout_count=confirmation_count,seed=seed+1,)
    validation_indices=first_split.heldout_indices[second_split.train_indices]
    confirmation_indices=first_split.heldout_indices[second_split.heldout_indices]
    confirmation_hash=bytes2hex(SHA.sha256(join(confirmation_indices,',')))
    return RealQueryPartitions(first_split.train_indices,validation_indices,confirmation_indices,confirmation_count,confirmation_hash,seed)
end

function proportional_allocation(total::Int,capacities::AbstractVector{<:Integer})
    total>=0||throw(ArgumentError("allocation total cannot be negative"))
    total<=sum(capacities)||throw(ArgumentError("allocation exceeds capacity"))
    iszero(total)&&return zeros(Int,length(capacities))
    raw=Float64[total*capacity/sum(capacities) for capacity in capacities]
    allocation=min.(floor.(Int,raw),Int.(capacities))
    remaining=total-sum(allocation)
    order=sortperm(eachindex(capacities);by=index->(raw[index]-allocation[index],capacities[index]-allocation[index]),rev=true,)

    while remaining>0
        changed=false
        for index in order
            allocation[index]>=capacities[index]&&continue
            allocation[index]+=1
            remaining-=1
            changed=true
            iszero(remaining)&&break
        end
        changed||throw(ArgumentError("allocation cannot satisfy capacity"))
    end
    return allocation
end

function balanced_query_partitions(labels::AbstractVector{<:Integer};development_count::Int=256,validation_count::Int=128,confirmation_count::Int=256,seed::Int=42,strata::AbstractVector{<:Integer},)
    counts=[development_count,validation_count,confirmation_count]
    all(>(0),counts)||throw(ArgumentError("partition counts must be positive"))
    length(strata)==length(labels)||throw(DimensionMismatch("query strata do not match labels"))
    sum(counts)<=length(labels)||throw(ArgumentError("not enough queries for partitions"))
    rng=MersenneTwister(seed)
    groups=Dict{Int,Vector{Int}}()
    for(index,stratum) in enumerate(strata)
        push!(get!(groups,Int(stratum),Int[]),index)
    end
    for group in values(groups)
        shuffle!(rng,group)
    end
    keys_order=sort!(collect(keys(groups)))
    partition_indices=[Int[] for _ in counts]
    target_per_stratum=floor.(Int,counts./length(keys_order))

    for stratum in keys_order
        group=groups[stratum]
        available=min(length(group),sum(target_per_stratum))
        allocation=available==sum(target_per_stratum) ? copy(target_per_stratum) : proportional_allocation(available,counts)
        position=1
        for partition in eachindex(counts)
            take=allocation[partition]
            append!(partition_indices[partition],group[position:position+take-1])
            position+=take
        end
        groups[stratum]=group[position:end]
    end

    remaining=Int[]
    for stratum in keys_order
        append!(remaining,groups[stratum])
    end
    shuffle!(rng,remaining)
    position=1
    for partition in eachindex(counts)
        needed=counts[partition]-length(partition_indices[partition])
        needed>=0||throw(ArgumentError("stratified allocation exceeded partition size"))
        append!(partition_indices[partition],remaining[position:position+needed-1])
        position+=needed
        shuffle!(rng,partition_indices[partition])
    end

    development_indices,validation_indices,confirmation_indices=partition_indices
    confirmation_hash=bytes2hex(SHA.sha256(join(confirmation_indices,',')))
    return RealQueryPartitions(development_indices,validation_indices,confirmation_indices,confirmation_count,confirmation_hash,seed)
end

function validate_query_partitions(partitions::RealQueryPartitions,query_count::Int)
    query_count>0||throw(ArgumentError("query count must be positive"))
    public_indices=vcat(partitions.development_indices,partitions.validation_indices)
    all(index->1<=index<=query_count,public_indices)||throw(BoundsError(1:query_count,public_indices))
    isempty(intersect(Set(partitions.development_indices),Set(partitions.validation_indices)))||throw(ArgumentError("development and validation queries overlap"))

    if !isempty(partitions.confirmation_indices)
        length(partitions.confirmation_indices)==partitions.confirmation_count||throw(DimensionMismatch("confirmation query count changed"))
        all(index->1<=index<=query_count,partitions.confirmation_indices)||throw(BoundsError(1:query_count,partitions.confirmation_indices))
        isempty(intersect(Set(public_indices),Set(partitions.confirmation_indices)))||throw(ArgumentError("confirmation queries overlap public partitions"))
        actual_hash=bytes2hex(SHA.sha256(join(partitions.confirmation_indices,',')))
        actual_hash==partitions.confirmation_hash||throw(ArgumentError("confirmation query hash changed"))
    end

    return partitions
end

function save_query_partitions(path::AbstractString,partitions::RealQueryPartitions;dataset_hash::AbstractString,query_count::Int,)
    isempty(partitions.confirmation_indices)&&throw(ArgumentError("confirmation indices are required when sealing partitions"))
    validate_query_partitions(partitions,query_count)
    mkpath(dirname(path))
    public_path="$(path).toml"
    sealed_path="$(path).confirmation.toml"
    public_data=Dict(
        "dataset_hash"=>String(dataset_hash),
        "query_count"=>query_count,
        "seed"=>partitions.seed,
        "development_indices"=>partitions.development_indices,
        "validation_indices"=>partitions.validation_indices,
        "confirmation_count"=>partitions.confirmation_count,
        "confirmation_hash"=>partitions.confirmation_hash,
    )
    sealed_data=Dict(
        "dataset_hash"=>String(dataset_hash),
        "confirmation_indices"=>partitions.confirmation_indices,
        "confirmation_hash"=>partitions.confirmation_hash,
    )
    open(public_path,"w") do io
        TOML.print(io,public_data)
    end
    open(sealed_path,"w") do io
        TOML.print(io,sealed_data)
    end
    return(public=public_path,sealed=sealed_path,)
end

function load_query_partitions(path::AbstractString;dataset_hash::AbstractString,allow_confirmation::Bool=false,)
    public_path="$(path).toml"
    isfile(public_path)||throw(ArgumentError("query partition manifest does not exist"))
    data=TOML.parsefile(public_path)
    String(data["dataset_hash"])==dataset_hash||throw(ArgumentError("query partitions belong to another dataset"))
    confirmation_indices=Int[]

    if allow_confirmation
        sealed_path="$(path).confirmation.toml"
        isfile(sealed_path)||throw(ArgumentError("sealed confirmation partition does not exist"))
        sealed=TOML.parsefile(sealed_path)
        String(sealed["dataset_hash"])==dataset_hash||throw(ArgumentError("confirmation partition belongs to another dataset"))
        String(sealed["confirmation_hash"])==String(data["confirmation_hash"])||throw(ArgumentError("confirmation partition hash changed"))
        confirmation_indices=Int.(sealed["confirmation_indices"])
    end

    partitions=RealQueryPartitions(
        Int.(data["development_indices"]),
        Int.(data["validation_indices"]),
        confirmation_indices,
        Int(data["confirmation_count"]),
        String(data["confirmation_hash"]),
        Int(data["seed"]),
    )
    return validate_query_partitions(partitions,Int(data["query_count"]))
end

function query_partition_indices(partitions::RealQueryPartitions,partition::Symbol;allow_confirmation::Bool=false,)
    partition===:development&&return copy(partitions.development_indices)
    partition===:validation&&return copy(partitions.validation_indices)
    partition===:confirmation||throw(ArgumentError("partition must be development, validation or confirmation"))
    allow_confirmation||throw(ArgumentError("confirmation partition is sealed"))
    isempty(partitions.confirmation_indices)&&throw(ArgumentError("confirmation partition was not loaded"))
    return copy(partitions.confirmation_indices)
end

function dataset_selectivities(metadata::AbstractVector,filters::AbstractVector)
    counts=Dict{Int,Int}()
    for row in metadata
        counts[row.label]=get(counts,row.label,0)+1
    end
    return[get(counts,filter.label,0)/length(metadata) for filter in filters]
end

function arxiv_manifest(path::AbstractString,scale::Symbol,seed::Int;verify_hashes::Bool=false,)
    vector_info=fvecs_info(joinpath(path,"database_vectors.fvecs"))
    query_info=fvecs_info(joinpath(path,"query_vectors.fvecs"))
    expected_count=ARXIV_FANNS_COUNTS[scale]
    vector_info.count==expected_count||throw(DimensionMismatch("database vector count does not match $(scale) dataset"))
    vector_info.dimension==ARXIV_FANNS_DIMENSION||throw(DimensionMismatch("database vector dimension does not match source"))
    query_info.count==ARXIV_FANNS_QUERY_COUNT||throw(DimensionMismatch("query vector count does not match source"))
    query_info.dimension==vector_info.dimension||throw(DimensionMismatch("query dimension does not match database"))
    manifest_path=joinpath(path,"sen_dataset.toml")

    if isfile(manifest_path)
        manifest=load_dataset_manifest(manifest_path)
        return validate_dataset_manifest(manifest,path;verify_hashes=verify_hashes,)
    end

    manifest=create_dataset_manifest(
        path;
        name="arxiv-for-fanns-$(scale)",
        source="https://huggingface.co/datasets/$(arxiv_fanns_repository(scale))",
        revision="main",
        license=ARXIV_FANNS_LICENSE,
        vector_count=vector_info.count,
        dimension=vector_info.dimension,
        query_count=query_info.count,
        metric=:cosine,
        sampling_seed=seed,
        filenames=ARXIV_FANNS_FILES,
    )
    save_dataset_manifest(manifest_path,manifest)
    return manifest
end

function load_arxiv_fanns_indices(path::AbstractString,train_indices::AbstractVector{<:Integer},heldout_indices::AbstractVector{<:Integer};scale::Symbol=:medium,k::Int=10,seed::Int=42,verify_hashes::Bool=false,)
    k>0||throw(ArgumentError("k must be positive"))
    manifest=arxiv_manifest(path,scale,seed;verify_hashes=verify_hashes,)
    metadata=load_arxiv_metadata(joinpath(path,"database_attributes.jsonl"))
    length(metadata)==manifest.vector_count||throw(DimensionMismatch("metadata count does not match vectors"))
    labels=load_arxiv_query_labels(joinpath(path,"em_query_attributes.jsonl"))
    length(labels)==manifest.query_count||throw(DimensionMismatch("query attribute count does not match queries"))
    resolved_train=Int.(train_indices)
    resolved_heldout=Int.(heldout_indices)
    isempty(resolved_train)&&throw(ArgumentError("train query indices cannot be empty"))
    isempty(resolved_heldout)&&throw(ArgumentError("heldout query indices cannot be empty"))
    all_indices=vcat(resolved_train,resolved_heldout)
    all(index->1<=index<=manifest.query_count,all_indices)||throw(BoundsError(1:manifest.query_count,all_indices))
    length(unique(all_indices))==length(all_indices)||throw(ArgumentError("query partitions cannot overlap"))
    all_queries=load_fvecs_indices(joinpath(path,"query_vectors.fvecs"),all_indices)
    all_truth=load_ivecs_indices(joinpath(path,"ground_truth_em.ivecs"),all_indices;k=k,)
    vectors=load_fvecs(joinpath(path,"database_vectors.fvecs"))
    filters=NamedTuple[(label=labels[index],) for index in all_indices]
    train_count=length(resolved_train)
    heldout_count=length(resolved_heldout)
    train_range=1:train_count
    heldout_range=train_count+1:train_count+heldout_count

    return ArxivFANNSDataset(
        manifest,
        String(path),
        vectors,
        metadata,
        all_queries[:,train_range],
        filters[train_range],
        all_truth[train_range],
        resolved_train,
        all_queries[:,heldout_range],
        filters[heldout_range],
        all_truth[heldout_range],
        resolved_heldout,
    )
end

function load_arxiv_fanns(path::AbstractString;scale::Symbol=:medium,train_count::Int=24,heldout_count::Int=48,k::Int=10,seed::Int=42,verify_hashes::Bool=false,)
    labels=load_arxiv_query_labels(joinpath(path,"em_query_attributes.jsonl"))
    split=stratified_query_split(labels;train_count=train_count,heldout_count=heldout_count,seed=seed,)
    return load_arxiv_fanns_indices(path,split.train_indices,split.heldout_indices;scale=scale,k=k,seed=seed,verify_hashes=verify_hashes,)
end

function verify_arxiv_groundtruth(dataset::ArxivFANNSDataset;sample_count::Int=1,k::Int=10,split::Symbol=:train,)
    sample_count>0||throw(ArgumentError("sample count must be positive"))
    k>0||throw(ArgumentError("k must be positive"))
    queries=split===:train ? dataset.train_queries : split===:heldout ? dataset.heldout_queries : throw(ArgumentError("split must be train or heldout"))
    filters=split===:train ? dataset.train_filters : dataset.heldout_filters
    truth=split===:train ? dataset.train_truth : dataset.heldout_truth
    count=min(sample_count,size(queries,2))
    exact=compute_groundtruth(dataset.vectors,dataset.metadata,queries[:,1:count];k=k,metric=:cosine,filters=filters[1:count],)
    recalls=[recall_at_k(exact[index],truth[index],k) for index in 1:count]
    return(passed=all(==(1.0),recalls),recalls=recalls,checked=count,)
end
