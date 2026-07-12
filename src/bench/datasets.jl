using Random

function   generate_synthetic_dataset(
    n::Int,
    dim::Int;
    seed::Int = 42,
)
    n>0 || throw(ArgumentError("n must be positive"))
    dim>0 || throw(ArgumentError("dim must be positive"))

    rng = MersenneTwister(seed)
    vectors = randn(rng, Float32,dim,n)

    for i in 1:n
        vector = @view vectors[:,i]
        vector ./= sqrt(sum(abs2,vector))
    end

    topics = ["systems","machine-learning","databases"]
    languages = ["julia","python","rust"]

    metadata = [
        (
        topic = rand(rng, topics),
        language = rand(rng, languages),
        year = rand(rng, 2020:2026),
        )
        for _ in 1:n
    ]

    return (
        vectors = vectors, metadata=metadata,
    )
        end