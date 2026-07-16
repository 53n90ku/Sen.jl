function validated_float32(value::Real, context::AbstractString, index)
    isfinite(value)||throw(ArgumentError("$context contains a non-finite value at $index"))

    converted=try
        Float32(value)
    catch error
        error isa InexactError||error isa OverflowError||rethrow()
        throw(
            ArgumentError(
                "$context contains a value that cannot be represented as Float32 at $index",
            ),
        )
    end

    isfinite(converted)||throw(
        ArgumentError("$context contains a value that overflows Float32 at $index"),
    )
    return converted
end

function validate_vector_values!(
    vector::AbstractVector{Float32},
    metric::Symbol,
    context::AbstractString,
)
    any_nonzero=false

    for (index, value) in pairs(vector)
        isfinite(value)||throw(
            ArgumentError("$context contains a non-finite value at $index"),
        )
        any_nonzero|=!iszero(value)
    end

    metric===:cosine&&!any_nonzero&&throw(
        ArgumentError("$context cannot be a zero vector for cosine similarity"),
    )
    return vector
end

function convert_validated_vector(
    vector::AbstractVector{<:Real},
    dim::Int,
    metric::Symbol;
    context::AbstractString = "vector",
)
    length(vector)==dim||throw(
        DimensionMismatch("$context dimension doesnt match database"),
    )
    converted=Vector{Float32}(undef, dim)

    for index = 1:dim
        converted[index]=validated_float32(vector[index], context, index)
    end

    return validate_vector_values!(converted, metric, context)
end

function convert_validated_vectors(
    vectors::AbstractMatrix{<:Real},
    dim::Int,
    metric::Symbol;
    context::AbstractString = "vector",
)
    size(vectors, 1)==dim||throw(
        DimensionMismatch("$(context) dimensions dont match database"),
    )
    converted=Matrix{Float32}(undef, size(vectors))

    for column in axes(vectors, 2)
        label="$context $column"

        for row = 1:dim
            converted[row, column]=validated_float32(vectors[row, column], label, row)
        end

        validate_vector_values!(@view(converted[:, column]), metric, label)
    end

    return converted
end

function validate_stored_vectors!(
    vectors::AbstractMatrix{Float32},
    metric::Symbol;
    context::AbstractString = "stored vector",
)
    for column in axes(vectors, 2)
        validate_vector_values!(@view(vectors[:, column]), metric, "$context $column")
    end

    return vectors
end
