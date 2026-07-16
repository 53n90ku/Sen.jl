using Dates
import Serialization

const PORTABLE_VALUE_FORMAT_VERSION=1
const PORTABLE_MAX_DIMENSIONS=8
const PORTABLE_MAX_CONTAINER_LENGTH=1_000_000_000
const PORTABLE_MAX_STRING_BYTES=1_000_000_000
const JULIA_SERIALIZATION_HEADER_PREFIX=UInt8[0x37, 0x4a, 0x4c]

const PORTABLE_MISSING=UInt8(0)
const PORTABLE_NOTHING=UInt8(1)
const PORTABLE_FALSE=UInt8(2)
const PORTABLE_TRUE=UInt8(3)
const PORTABLE_INT64=UInt8(4)
const PORTABLE_UINT64=UInt8(5)
const PORTABLE_FLOAT32=UInt8(6)
const PORTABLE_FLOAT64=UInt8(7)
const PORTABLE_STRING=UInt8(8)
const PORTABLE_SYMBOL=UInt8(9)
const PORTABLE_DATE=UInt8(10)
const PORTABLE_DATETIME=UInt8(11)
const PORTABLE_ARRAY=UInt8(12)

function write_portable_uint16(io::IO, value::Integer)
    write(io, htol(UInt16(value)))
    return io
end

function read_portable_uint16(io::IO)
    return ltoh(read(io, UInt16))
end

function write_portable_uint32(io::IO, value::Integer)
    write(io, htol(UInt32(value)))
    return io
end

function read_portable_uint32(io::IO)
    return ltoh(read(io, UInt32))
end

function write_portable_uint64(io::IO, value::Integer)
    write(io, htol(UInt64(value)))
    return io
end

function read_portable_uint64(io::IO)
    return ltoh(read(io, UInt64))
end

function write_portable_int64(io::IO, value::Signed)
    converted=Int64(value)
    write_portable_uint64(io, reinterpret(UInt64, converted))
    return io
end

function read_portable_int64(io::IO)
    return reinterpret(Int64, read_portable_uint64(io))
end

function write_portable_float32(io::IO, value::Real)
    write_portable_uint32(io, reinterpret(UInt32, Float32(value)))
    return io
end

function read_portable_float32(io::IO)
    return reinterpret(Float32, read_portable_uint32(io))
end

function write_portable_float64(io::IO, value::Real)
    write_portable_uint64(io, reinterpret(UInt64, Float64(value)))
    return io
end

function read_portable_float64(io::IO)
    return reinterpret(Float64, read_portable_uint64(io))
end

function write_portable_length(io::IO, value::Integer, name::AbstractString)
    value>=0||throw(ArgumentError("$(name) cannot be negative"))
    value<=PORTABLE_MAX_CONTAINER_LENGTH||throw(ArgumentError("$(name) is too large"))
    write_portable_uint64(io, value)
    return io
end

function read_portable_length(
    io::IO,
    name::AbstractString;
    maximum::Int = PORTABLE_MAX_CONTAINER_LENGTH,
)
    value=read_portable_uint64(io)
    value<=UInt64(maximum)||throw(ArgumentError("stored $(name) is too large"))
    return Int(value)
end

function write_portable_text(io::IO, value::AbstractString)
    text=String(value)
    isvalid(text)||throw(ArgumentError("portable strings must contain valid UTF-8"))
    bytes=codeunits(text)
    length(bytes)<=PORTABLE_MAX_STRING_BYTES||throw(
        ArgumentError("portable string is too large"),
    )
    write_portable_uint64(io, length(bytes))
    write(io, bytes)
    return io
end

function read_portable_text(io::IO)
    length=read_portable_length(io, "string length"; maximum = PORTABLE_MAX_STRING_BYTES)
    text=String(read(io, length))
    isvalid(text)||throw(ArgumentError("stored string is not valid UTF-8"))
    return text
end

function portable_array_type(array::AbstractArray)
    eltype(array)===Bool&&return PORTABLE_TRUE
    eltype(array)<:Signed&&return PORTABLE_INT64
    eltype(array)<:Unsigned&&return PORTABLE_UINT64
    eltype(array)===Float32&&return PORTABLE_FLOAT32
    eltype(array)<:AbstractFloat&&return PORTABLE_FLOAT64
    eltype(array)<:AbstractString&&return PORTABLE_STRING
    eltype(array)===Symbol&&return PORTABLE_SYMBOL
    eltype(array)===Date&&return PORTABLE_DATE
    eltype(array)===DateTime&&return PORTABLE_DATETIME
    throw(ArgumentError("unsupported portable array element type $(eltype(array))"))
end

function write_portable_array_element(io::IO, type::UInt8, value)
    if type===PORTABLE_TRUE
        write(io, value ? UInt8(1) : UInt8(0))
    elseif type===PORTABLE_INT64
        write_portable_int64(io, value)
    elseif type===PORTABLE_UINT64
        write_portable_uint64(io, value)
    elseif type===PORTABLE_FLOAT32
        write_portable_float32(io, value)
    elseif type===PORTABLE_FLOAT64
        write_portable_float64(io, value)
    elseif type===PORTABLE_STRING
        write_portable_text(io, value)
    elseif type===PORTABLE_SYMBOL
        write_portable_text(io, String(value))
    elseif type===PORTABLE_DATE||type===PORTABLE_DATETIME
        write_portable_text(io, string(value))
    else
        throw(ArgumentError("unsupported portable array type tag"))
    end

    return io
end

function write_portable_array(io::IO, array::AbstractArray)
    dimensions=ndims(array)
    1<=dimensions<=PORTABLE_MAX_DIMENSIONS||throw(
        ArgumentError(
            "portable arrays must have between 1 and $(PORTABLE_MAX_DIMENSIONS) dimensions",
        ),
    )
    length(array)<=PORTABLE_MAX_CONTAINER_LENGTH||throw(
        ArgumentError("portable array is too large"),
    )
    type=portable_array_type(array)
    write(io, type)
    write(io, UInt8(dimensions))

    for dimension in size(array)
        write_portable_length(io, dimension, "array dimension")
    end

    for value in array
        write_portable_array_element(io, type, value)
    end

    return io
end

function read_portable_array_element(io::IO, type::UInt8)
    if type===PORTABLE_TRUE
        value=read(io, UInt8)
        value in (UInt8(0), UInt8(1))||throw(
            ArgumentError("stored boolean array value is invalid"),
        )
        return value==UInt8(1)
    end

    type===PORTABLE_INT64&&return read_portable_int64(io)
    type===PORTABLE_UINT64&&return read_portable_uint64(io)
    type===PORTABLE_FLOAT32&&return read_portable_float32(io)
    type===PORTABLE_FLOAT64&&return read_portable_float64(io)
    type===PORTABLE_STRING&&return read_portable_text(io)
    type===PORTABLE_SYMBOL&&return Symbol(read_portable_text(io))
    type===PORTABLE_DATE&&return Date(read_portable_text(io))
    type===PORTABLE_DATETIME&&return DateTime(read_portable_text(io))
    throw(ArgumentError("unsupported stored portable array type tag"))
end

function portable_array_element_type(type::UInt8)
    type===PORTABLE_TRUE&&return Bool
    type===PORTABLE_INT64&&return Int64
    type===PORTABLE_UINT64&&return UInt64
    type===PORTABLE_FLOAT32&&return Float32
    type===PORTABLE_FLOAT64&&return Float64
    type===PORTABLE_STRING&&return String
    type===PORTABLE_SYMBOL&&return Symbol
    type===PORTABLE_DATE&&return Date
    type===PORTABLE_DATETIME&&return DateTime
    throw(ArgumentError("unsupported stored portable array type tag"))
end

function read_portable_array(io::IO)
    type=read(io, UInt8)
    element_type=portable_array_element_type(type)
    dimensions=Int(read(io, UInt8))
    1<=dimensions<=PORTABLE_MAX_DIMENSIONS||throw(
        ArgumentError("stored portable array dimension count is invalid"),
    )
    shape=Vector{Int}(undef, dimensions)
    count=1

    for index in eachindex(shape)
        shape[index]=read_portable_length(io, "array dimension")
        count=Base.checked_mul(count, shape[index])
        count<=PORTABLE_MAX_CONTAINER_LENGTH||throw(
            ArgumentError("stored portable array is too large"),
        )
    end

    array=Array{element_type}(undef, Tuple(shape))

    for index in eachindex(array)
        array[index]=read_portable_array_element(io, type)
    end

    return array
end

function write_portable_value(io::IO, value)
    if value===missing
        write(io, PORTABLE_MISSING)
    elseif value===nothing
        write(io, PORTABLE_NOTHING)
    elseif value===false
        write(io, PORTABLE_FALSE)
    elseif value===true
        write(io, PORTABLE_TRUE)
    elseif value isa Signed
        write(io, PORTABLE_INT64)
        write_portable_int64(io, value)
    elseif value isa Unsigned
        write(io, PORTABLE_UINT64)
        write_portable_uint64(io, value)
    elseif value isa Float32
        write(io, PORTABLE_FLOAT32)
        write_portable_float32(io, value)
    elseif value isa AbstractFloat
        write(io, PORTABLE_FLOAT64)
        write_portable_float64(io, value)
    elseif value isa AbstractString
        write(io, PORTABLE_STRING)
        write_portable_text(io, value)
    elseif value isa Symbol
        write(io, PORTABLE_SYMBOL)
        write_portable_text(io, String(value))
    elseif value isa Date
        write(io, PORTABLE_DATE)
        write_portable_text(io, string(value))
    elseif value isa DateTime
        write(io, PORTABLE_DATETIME)
        write_portable_text(io, string(value))
    elseif value isa AbstractArray
        write(io, PORTABLE_ARRAY)
        write_portable_array(io, value)
    else
        throw(ArgumentError("unsupported portable value type $(typeof(value))"))
    end

    return io
end

function read_portable_value(io::IO)
    type=read(io, UInt8)
    type===PORTABLE_MISSING&&return missing
    type===PORTABLE_NOTHING&&return nothing
    type===PORTABLE_FALSE&&return false
    type===PORTABLE_TRUE&&return true
    type===PORTABLE_INT64&&return read_portable_int64(io)
    type===PORTABLE_UINT64&&return read_portable_uint64(io)
    type===PORTABLE_FLOAT32&&return read_portable_float32(io)
    type===PORTABLE_FLOAT64&&return read_portable_float64(io)
    type===PORTABLE_STRING&&return read_portable_text(io)
    type===PORTABLE_SYMBOL&&return Symbol(read_portable_text(io))
    type===PORTABLE_DATE&&return Date(read_portable_text(io))
    type===PORTABLE_DATETIME&&return DateTime(read_portable_text(io))
    type===PORTABLE_ARRAY&&return read_portable_array(io)
    throw(ArgumentError("unsupported stored portable value type tag"))
end

function write_portable_named_tuple(io::IO, value::NamedTuple)
    names=propertynames(value)
    write_portable_length(io, length(names), "metadata field count")

    for name in names
        write_portable_text(io, String(name))
        write_portable_value(io, getproperty(value, name))
    end

    return io
end

function read_portable_named_tuple(io::IO)
    count=read_portable_length(io, "metadata field count")
    names=Vector{Symbol}(undef, count)
    values=Vector{Any}(undef, count)
    seen=Set{Symbol}()

    for index = 1:count
        name=Symbol(read_portable_text(io))
        name in seen&&throw(ArgumentError("stored metadata contains duplicate fields"))
        push!(seen, name)
        names[index]=name
        values[index]=read_portable_value(io)
    end

    return NamedTuple{Tuple(names)}(Tuple(values))
end

function is_legacy_julia_serialization_header(bytes::AbstractVector{UInt8})
    return length(bytes)>=length(JULIA_SERIALIZATION_HEADER_PREFIX)&&bytes[1:length(
        JULIA_SERIALIZATION_HEADER_PREFIX,
    )]==JULIA_SERIALIZATION_HEADER_PREFIX
end

function portable_read_error(error, message::AbstractString)
    error isa InterruptException&&rethrow(error)
    error isa ArgumentError&&rethrow(error)
    error isa EOFError&&throw(ArgumentError("$(message) is truncated"))
    error isa InexactError&&throw(
        ArgumentError("$(message) contains an out-of-range value"),
    )
    error isa OverflowError&&throw(ArgumentError("$(message) contains invalid dimensions"))
    rethrow(error)
end
