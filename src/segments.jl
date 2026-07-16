struct ImmutableSegment
    id::String
    revision_start::UInt64
    revision_end::UInt64
    vector_store::VectorStore
    metadata_store::MetadataStore
    id_store::IDStore
    excluded::BitVector
    tombstone_ids::Set{Any}
    index::Union{Nothing,FilterAwareIVFIndex}
    filter_index::Union{Nothing,BitsetIndex}
    build_config::Union{Nothing,IndexBuildConfig}
    index_bytes::Int
end

mutable struct ActiveSegment
    id::String
    revision_start::UInt64
    revision_end::UInt64
    store::DeltaStore
    tombstone_ids::Set{Any}
end

function next_segment_revision(revision::UInt64)
    revision==typemax(UInt64)&&return revision
    return revision+UInt64(1)
end

function create_immutable_segment(
    id::AbstractString,
    revision_start::UInt64,
    revision_end::UInt64,
    vector_store::VectorStore,
    metadata_store::MetadataStore,
    id_store::IDStore;
    excluded::BitVector=falses(length(vector_store)),
    tombstone_ids::Set{Any}=Set{Any}(),
    index::Union{Nothing,FilterAwareIVFIndex}=nothing,
    filter_index::Union{Nothing,BitsetIndex}=nothing,
    build_config::Union{Nothing,IndexBuildConfig}=nothing,
    index_bytes::Int=index===nothing ? (filter_index===nothing ? 0 : Base.summarysize(filter_index)) : (filter_index===nothing ? 0 : Base.summarysize((index,filter_index,))),
)
    segment=ImmutableSegment(
        String(id),
        revision_start,
        revision_end,
        vector_store,
        metadata_store,
        id_store,
        excluded,
        tombstone_ids,
        index,
        filter_index,
        build_config,
        index_bytes,
    )
    return validate_immutable_segment(segment)
end

function create_active_segment(dim::Int,revision::UInt64;id::AbstractString="active-$(next_segment_revision(revision))",)
    start=next_segment_revision(revision)
    return ActiveSegment(String(id),start,revision,create_delta_store(dim),Set{Any}())
end

function validate_immutable_segment_fast(segment::ImmutableSegment)
    isempty(segment.id)&&throw(ArgumentError("segment id cannot be empty"))
    segment.revision_start<=segment.revision_end||throw(ArgumentError("immutable segment revision range is invalid"))
    count=length(segment.vector_store)
    count==length(segment.metadata_store)==length(segment.id_store)||error("immutable segment stores are misaligned")
    length(segment.excluded)==count||error("immutable segment exclusions are misaligned")

    if segment.index===nothing
        if segment.filter_index===nothing
            segment.index_bytes==0||error("immutable segment without indexes reports index bytes")
        else
            segment.filter_index.count==count||error("immutable segment filter index is misaligned")
            segment.index_bytes>0||error("metadata-indexed immutable segment reports no index bytes")
        end
    else
        segment.filter_index===nothing&&error("indexed immutable segment has no filter index")
        sum(length,segment.index.ivf.lists)==count||error("immutable segment vector index is misaligned")
        segment.filter_index.count==count||error("immutable segment filter index is misaligned")
        segment.index_bytes>0||error("indexed immutable segment reports no index bytes")
    end

    return segment
end

function validate_immutable_segment(segment::ImmutableSegment)
    validate_immutable_segment_fast(segment)
    !isempty(segment.tombstone_ids)&&any(id->id in segment.tombstone_ids,segment.id_store.ids)&&error("immutable segment records and tombstones overlap")
    return segment
end

function validate_active_segment_fast(segment::ActiveSegment,dim::Int,current_revision::UInt64)
    isempty(segment.id)&&throw(ArgumentError("active segment id cannot be empty"))
    segment.store.vector_store.dim==dim||throw(DimensionMismatch("active segment dimension is invalid"))
    validate_delta_store(segment.store)
    segment.revision_end<=current_revision||error("active segment revision is ahead of database")

    if isempty(segment.store.id_store.ids)&&isempty(segment.tombstone_ids)
        segment.revision_start==next_segment_revision(segment.revision_end)||error("empty active segment revision range is invalid")
    else
        segment.revision_start<=segment.revision_end||error("active segment revision range is invalid")
    end

    return segment
end


function validate_active_segment(segment::ActiveSegment,dim::Int,current_revision::UInt64)
    validate_active_segment_fast(segment,dim,current_revision)
    !isempty(segment.tombstone_ids)&&any(id->id in segment.tombstone_ids,segment.store.id_store.ids)&&error("active segment records and tombstones overlap")
    return segment
end

function immutable_segment_record(segment::ImmutableSegment,id)
    id in segment.tombstone_ids&&throw(KeyError(id))
    has_id(segment.id_store,id)||throw(KeyError(id))
    position=get_position(segment.id_store,id)
    segment.excluded[position]&&throw(KeyError(id))
    return(
        id=get_id(segment.id_store,position),
        vector=collect(get_vector(segment.vector_store,position)),
        metadata=get_metadata(segment.metadata_store,position),
    )
end

function immutable_segment_has_record(segment::ImmutableSegment,id)
    id in segment.tombstone_ids&&return false
    has_id(segment.id_store,id)||return false
    return !segment.excluded[get_position(segment.id_store,id)]
end

function active_segment_is_empty(segment::ActiveSegment)
    return isempty(segment.store.id_store.ids)&&isempty(segment.tombstone_ids)
end

function active_segment_work(segment::ActiveSegment)
    return length(segment.store)+length(segment.tombstone_ids)
end

function segment_topology_visible_ids(segments::Vector{ImmutableSegment},active::ActiveSegment)
    seen=Set{Any}(active.tombstone_ids)
    visible=Set{Any}()

    for id in active.store.id_store.ids
        id in seen&&continue
        push!(seen,id)
        push!(visible,id)
    end

    for segment in Iterators.reverse(segments)
        union!(seen,segment.tombstone_ids)

        for(position,id) in enumerate(segment.id_store.ids)
            id in seen&&continue
            push!(seen,id)
            segment.excluded[position]||push!(visible,id)
        end
    end

    return visible
end

function mark_active_segment_revision!(segment::ActiveSegment,revision::UInt64)
    if active_segment_is_empty(segment)&&segment.revision_start>segment.revision_end
        segment.revision_start=revision
    end

    segment.revision_end=revision
    return segment
end

function reset_active_segment!(segment::ActiveSegment,dim::Int,revision::UInt64)
    replacement=create_active_segment(dim,revision)
    segment.id=replacement.id
    segment.revision_start=replacement.revision_start
    segment.revision_end=replacement.revision_end
    segment.store=replacement.store
    segment.tombstone_ids=replacement.tombstone_ids
    return segment
end
