abstract type StoredSelection end

struct DenseStoredSelection <: StoredSelection
    mask::BitVector
    count::Int
end

struct SparseStoredSelection <: StoredSelection
    positions::Vector{Int}
end

abstract type OrderedRangeLane end

struct RangeLane{T} <: OrderedRangeLane
    values::Vector{T}
    positions::Vector{Int}
end

mutable struct RangeLaneBuilder{T}
    values::Vector{T}
    positions::Vector{Int}
end

struct MetadataIndex
    selections::Dict{Tuple{Symbol,Any},StoredSelection}
    range_lanes::Dict{Tuple{Symbol,Symbol},Vector{OrderedRangeLane}}
    count::Int
end

const BitsetIndex=MetadataIndex

struct TemporalSelectionKey{T<:Union{Date,DateTime}}
    value::T
end

selection_value_key(value::Union{Date,DateTime}) = TemporalSelectionKey(value)
selection_value_key(value) = value

selection_count(selection::DenseStoredSelection) = selection.count
selection_count(selection::SparseStoredSelection) = length(selection.positions)

function range_index_domain(value)
    value isa Bool&&return nothing
    value isa Date&&return :date
    value isa DateTime&&return :datetime
    value isa Real||return nothing
    value isa AbstractFloat&&isnan(value)&&return nothing
    return :numeric
end

range_less(left, right) = (left<right)===true

function push_range_value!(
    builders::Dict{Tuple{Symbol,Symbol,DataType},Any},
    field::Symbol,
    value,
    position::Int,
)
    domain=range_index_domain(value)
    domain===nothing&&return builders
    key=(field, domain, typeof(value))
    builder=get!(builders, key) do
        RangeLaneBuilder(typeof(value)[], Int[])
    end
    push!(builder.values, value)
    push!(builder.positions, position)
    return builders
end

function build_range_lanes(builders::Dict{Tuple{Symbol,Symbol,DataType},Any})
    range_lanes=Dict{Tuple{Symbol,Symbol},Vector{OrderedRangeLane}}()

    for ((field, domain, _), builder) in builders
        order=sortperm(builder.values; lt = range_less)
        lane=RangeLane(builder.values[order], builder.positions[order])
        push!(get!(range_lanes, (field, domain), OrderedRangeLane[]), lane)
    end

    return range_lanes
end

function build_stored_selection(positions::Vector{Int}, count::Int)
    dense_bytes=cld(count, 8)
    sparse_bytes=sizeof(Int)*length(positions)

    if dense_bytes<=sparse_bytes
        mask=falses(count)
        mask[positions].=true
        return DenseStoredSelection(mask, length(positions))
    end

    return SparseStoredSelection(positions)
end

function build_metadata_index(metadata::AbstractVector)
    count=length(metadata)
    postings=Dict{Tuple{Symbol,Any},Vector{Int}}()
    range_builders=Dict{Tuple{Symbol,Symbol,DataType},Any}()

    for (position, row) in enumerate(metadata)
        for (field, value) in pairs(row)
            push!(get!(postings, (field, selection_value_key(value)), Int[]), position)
            push_range_value!(range_builders, field, value, position)
        end
    end

    selections=Dict{Tuple{Symbol,Any},StoredSelection}()

    for (key, positions) in postings
        selections[key]=build_stored_selection(positions, count)
    end

    return MetadataIndex(selections, build_range_lanes(range_builders), count)
end

build_bitset_index(metadata::AbstractVector) = build_metadata_index(metadata)

function selection_mask(selection::DenseStoredSelection, count::Int)
    length(selection.mask)==count||throw(
        DimensionMismatch("selection count doesnt match metadata index"),
    )
    return copy(selection.mask)
end

function selection_mask(selection::SparseStoredSelection, count::Int)
    mask=falses(count)
    mask[selection.positions].=true
    return mask
end

function equality_filter_mask(index::MetadataIndex, field::Symbol, value)
    selection=get(index.selections, (field, selection_value_key(value)), nothing)
    return selection===nothing ? falses(index.count) :
           selection_mask(selection, index.count)
end

evaluate_filter(index::MetadataIndex, filter::NamedTuple) =
    evaluate_filter(index, normalize_filter(filter))
evaluate_filter(index::MetadataIndex, ::Nothing) = trues(index.count)
evaluate_filter(index::MetadataIndex, filter::Eq) =
    equality_filter_mask(index, filter.field, filter.value)

function evaluate_filter(index::MetadataIndex, filter::In)
    result=falses(index.count)

    for value in filter.values
        result.|=equality_filter_mask(index, filter.field, value)
    end

    return result
end

function evaluate_filter(index::MetadataIndex, filter::And)
    result=trues(index.count)

    for child in filter.children
        result.&=evaluate_filter(index, child)
        any(result)||break
    end

    return result
end

function evaluate_filter(index::MetadataIndex, filter::Or)
    result=falses(index.count)

    for child in filter.children
        result.|=evaluate_filter(index, child)
        all(result)&&break
    end

    return result
end

evaluate_filter(index::MetadataIndex, filter::Not) = .!evaluate_filter(index, filter.child)

function add_range_matches!(mask::BitVector, lane::RangeLane, lower, upper)
    first_index=searchsortedfirst(lane.values, lower; lt = range_less)
    last_index=searchsortedlast(lane.values, upper; lt = range_less)

    @inbounds for sorted_index = first_index:last_index
        mask[lane.positions[sorted_index]]=true
    end

    return mask
end

function evaluate_filter(index::MetadataIndex, filter::Range)
    result=falses(index.count)
    domain=range_domain(filter.lower)
    lanes=get(index.range_lanes, (filter.field, domain), nothing)
    lanes===nothing&&return result

    for lane in lanes
        add_range_matches!(result, lane, filter.lower, filter.upper)
    end

    return result
end

supports_indexed_filter(::MetadataIndex, ::Nothing) = true
supports_indexed_filter(::MetadataIndex, ::Eq) = true
supports_indexed_filter(::MetadataIndex, ::In) = true
supports_indexed_filter(::MetadataIndex, ::Range) = true
supports_indexed_filter(index::MetadataIndex, filter::And) =
    all(child->supports_indexed_filter(index, child), filter.children)
supports_indexed_filter(index::MetadataIndex, filter::Or) =
    all(child->supports_indexed_filter(index, child), filter.children)
supports_indexed_filter(index::MetadataIndex, filter::Not) =
    supports_indexed_filter(index, filter.child)
supports_indexed_filter(index::MetadataIndex, filter::NamedTuple) =
    supports_indexed_filter(index, normalize_filter(filter))
