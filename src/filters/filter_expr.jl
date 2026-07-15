const MAX_FILTER_DEPTH=64
const MAX_FILTER_NODES=4_096
const MAX_FILTER_IN_VALUES=4_096

abstract type FilterExpr end

function freeze_filter_value(value)
    value isa AbstractString&&return String(value)
    value===missing&&return missing
    value===nothing&&return nothing
    value isa Union{Bool,Integer,AbstractFloat,Symbol,Date,DateTime}&&return value
    throw(ArgumentError("filter values must be immutable scalar metadata values"))
end

function filter_value_isequal(left,right)::Bool
    left_temporal=left isa Union{Date,DateTime}
    right_temporal=right isa Union{Date,DateTime}
    (left_temporal||right_temporal)&&return typeof(left)===typeof(right)&&isequal(left,right)
    return isequal(left,right)
end

function filter_value_hash(value,seed::UInt)
    value isa Date&&return hash(value,hash(:date,seed))
    value isa DateTime&&return hash(value,hash(:datetime,seed))
    return hash(value,seed)
end

struct Eq{T} <: FilterExpr
    field::Symbol
    value::T

    function Eq(field::Symbol,value)
        frozen=freeze_filter_value(value)
        return new{typeof(frozen)}(field,frozen)
    end
end

Eq(field,value)=throw(ArgumentError("filter field must be a Symbol"))

function freeze_in_values(values)
    values isa AbstractString&&throw(ArgumentError("In values must be a collection"))
    frozen=Any[]
    value_count=0

    try
        for value in values
            value_count+=1
            value_count<=MAX_FILTER_IN_VALUES||throw(ArgumentError("In cannot contain more than $(MAX_FILTER_IN_VALUES) values"))
            candidate=freeze_filter_value(value)
            any(existing->filter_value_isequal(existing,candidate),frozen)&&continue
            push!(frozen,candidate)
        end
    catch error
        error isa ArgumentError&&rethrow()
        error isa MethodError&&throw(ArgumentError("In values must be a collection"))
        rethrow()
    end

    return Tuple(frozen)
end

struct In{T<:Tuple} <: FilterExpr
    field::Symbol
    values::T

    function In(field::Symbol,values)
        frozen=freeze_in_values(values)
        return new{typeof(frozen)}(field,frozen)
    end
end

In(field,values)=throw(ArgumentError("filter field must be a Symbol"))

function range_domain(value)
    value isa Bool&&return nothing
    value isa Date&&return :date
    value isa DateTime&&return :datetime
    value isa Real&&return :numeric
    return nothing
end

function validate_range_bounds(lower,upper)
    lower=freeze_filter_value(lower)
    upper=freeze_filter_value(upper)
    lower_domain=range_domain(lower)
    upper_domain=range_domain(upper)
    lower_domain===nothing&&throw(ArgumentError("Range bounds must be numeric, Date, or DateTime values"))
    lower_domain===upper_domain||throw(ArgumentError("Range bounds must use compatible domains"))
    lower isa AbstractFloat&&isnan(lower)&&throw(ArgumentError("Range bounds cannot be NaN"))
    upper isa AbstractFloat&&isnan(upper)&&throw(ArgumentError("Range bounds cannot be NaN"))

    ordered=try
        lower<=upper
    catch
        throw(ArgumentError("Range bounds must be comparable"))
    end

    ordered===true||throw(ArgumentError("Range lower bound cannot exceed upper bound"))
    return lower,upper
end

struct Range{L,U} <: FilterExpr
    field::Symbol
    lower::L
    upper::U

    function Range(field::Symbol,lower,upper)
        frozen_lower,frozen_upper=validate_range_bounds(lower,upper)
        return new{typeof(frozen_lower),typeof(frozen_upper)}(field,frozen_lower,frozen_upper)
    end
end

Range(field,lower,upper)=throw(ArgumentError("filter field must be a Symbol"))

struct And{T<:Tuple} <: FilterExpr
    children::T

    function And(children::Tuple)
        all(child->child isa FilterExpr,children)||throw(ArgumentError("And children must be filter expressions"))
        return new{typeof(children)}(children)
    end
end

And(children::FilterExpr...)=And(children)

struct Or{T<:Tuple} <: FilterExpr
    children::T

    function Or(children::Tuple)
        all(child->child isa FilterExpr,children)||throw(ArgumentError("Or children must be filter expressions"))
        return new{typeof(children)}(children)
    end
end

Or(children::FilterExpr...)=Or(children)

struct Not{T<:FilterExpr} <: FilterExpr
    child::T
end

Not(child)=throw(ArgumentError("Not child must be a filter expression"))

function validate_filter(filter::FilterExpr)
    pending=Tuple{FilterExpr,Int}[(filter,1)]
    node_count=0

    while !isempty(pending)
        current,depth=pop!(pending)
        depth<=MAX_FILTER_DEPTH||throw(ArgumentError("filter depth cannot exceed $(MAX_FILTER_DEPTH)"))
        node_count+=1
        node_count<=MAX_FILTER_NODES||throw(ArgumentError("filter cannot contain more than $(MAX_FILTER_NODES) nodes"))

        if current isa And||current isa Or
            for child in current.children
                push!(pending,(child,depth+1))
            end
        elseif current isa Not
            push!(pending,(current.child,depth+1))
        elseif current isa In
            length(current.values)<=MAX_FILTER_IN_VALUES||throw(ArgumentError("In cannot contain more than $(MAX_FILTER_IN_VALUES) values"))
        elseif !(current isa Eq||current isa Range)
            throw(ArgumentError("unsupported filter expression $(typeof(current))"))
        end
    end

    return filter
end

normalize_filter(::Nothing)=nothing
normalize_filter(filter::FilterExpr)=validate_filter(filter)

function normalize_filter(filter::NamedTuple)
    children=Tuple(Eq(field,value) for(field,value) in pairs(filter))
    return validate_filter(And(children))
end

function range_contains(filter::Range,value)::Bool
    value===missing&&return false
    value isa Bool&&return false
    value isa AbstractFloat&&isnan(value)&&return false
    value_domain=range_domain(value)
    bound_domain=range_domain(filter.lower)
    value_domain===bound_domain||return false

    try
        return (filter.lower<=value)===true&&(value<=filter.upper)===true
    catch
        return false
    end
end

function matches_filter(metadata::NamedTuple,filter::Eq)::Bool
    hasproperty(metadata,filter.field)||return false
    return filter_value_isequal(getproperty(metadata,filter.field),filter.value)
end

function matches_filter(metadata::NamedTuple,filter::In)::Bool
    hasproperty(metadata,filter.field)||return false
    actual=getproperty(metadata,filter.field)
    return any(value->filter_value_isequal(actual,value),filter.values)
end

function matches_filter(metadata::NamedTuple,filter::Range)::Bool
    hasproperty(metadata,filter.field)||return false
    return range_contains(filter,getproperty(metadata,filter.field))
end

matches_filter(metadata::NamedTuple,filter::And)::Bool=all(child->matches_filter(metadata,child),filter.children)
matches_filter(metadata::NamedTuple,filter::Or)::Bool=any(child->matches_filter(metadata,child),filter.children)
matches_filter(metadata::NamedTuple,filter::Not)::Bool=!matches_filter(metadata,filter.child)
matches_filter(metadata::NamedTuple,filter::NamedTuple)::Bool=matches_filter(metadata,normalize_filter(filter))

Base.isequal(::FilterExpr,::FilterExpr)=false
Base.isequal(left::Eq,right::Eq)=left.field===right.field&&filter_value_isequal(left.value,right.value)

function Base.isequal(left::In,right::In)
    left.field===right.field||return false
    length(left.values)==length(right.values)||return false
    return all(filter_value_isequal(left.values[index],right.values[index]) for index in eachindex(left.values))
end

Base.isequal(left::Range,right::Range)=left.field===right.field&&filter_value_isequal(left.lower,right.lower)&&filter_value_isequal(left.upper,right.upper)
Base.isequal(left::And,right::And)=isequal(left.children,right.children)
Base.isequal(left::Or,right::Or)=isequal(left.children,right.children)
Base.isequal(left::Not,right::Not)=isequal(left.child,right.child)
Base.:(==)(left::FilterExpr,right::FilterExpr)::Bool=isequal(left,right)

Base.hash(filter::Eq,seed::UInt)=filter_value_hash(filter.value,hash(filter.field,hash(:Eq,seed)))

function Base.hash(filter::In,seed::UInt)
    result=hash(filter.field,hash(:In,seed))

    for value in filter.values
        result=filter_value_hash(value,result)
    end

    return result
end

Base.hash(filter::Range,seed::UInt)=filter_value_hash(filter.upper,filter_value_hash(filter.lower,hash(filter.field,hash(:Range,seed))))
Base.hash(filter::And,seed::UInt)=hash(filter.children,hash(:And,seed))
Base.hash(filter::Or,seed::UInt)=hash(filter.children,hash(:Or,seed))
Base.hash(filter::Not,seed::UInt)=hash(filter.child,hash(:Not,seed))
