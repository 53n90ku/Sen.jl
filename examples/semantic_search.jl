using Downloads
using JSON3
using Sen

const DEFAULT_EMBEDDING_MODEL="all-minilm"
const DEFAULT_OLLAMA_URL="http://127.0.0.1:11434/api/embed"

const SAMPLE_DOCUMENTS=[
    (
        id = "voyager-1",
        title = "Voyager 1's Interstellar Journey",
        category = "space",
        text = "Launched in 1977, Voyager 1 crossed into interstellar space in 2012, becoming the first human-made object to leave the heliosphere. It continues to send data back using a radioisotope power source.",
    ),
    (
        id = "james-webb",
        title = "The James Webb Space Telescope",
        category = "space",
        text = "JWST observes primarily in infrared, allowing it to peer through dust clouds and see some of the earliest galaxies formed after the Big Bang. It orbits the Sun near the L2 Lagrange point.",
    ),
    (
        id = "perseverance",
        title = "Mars Rover Perseverance",
        category = "space",
        text = "Perseverance landed in Jezero Crater in 2021, searching for signs of ancient microbial life and collecting rock samples for a future return mission to Earth.",
    ),
    (
        id = "black-hole-imaging",
        title = "Black Hole Imaging",
        category = "space",
        text = "In 2019, the Event Horizon Telescope produced the first direct image of a black hole's shadow, located in the galaxy M87, by combining data from radio telescopes worldwide.",
    ),
    (
        id = "solar-flares",
        title = "Solar Flares",
        category = "space",
        text = "Solar flares are sudden bursts of radiation from the Sun's surface. Strong flares can disrupt satellite communications and power grids on Earth.",
    ),
    (
        id = "exoplanet-detection",
        title = "Exoplanet Detection Methods",
        category = "space",
        text = "Astronomers primarily find exoplanets using the transit method, which detects the dimming of a star's light as a planet passes in front of it.",
    ),
    (
        id = "international-space-station",
        title = "The International Space Station",
        category = "space",
        text = "The ISS has been continuously inhabited since November 2000, serving as a microgravity research lab jointly operated by NASA, Roscosmos, ESA, JAXA, and CSA.",
    ),
    (
        id = "neutron-stars",
        title = "Neutron Stars",
        category = "space",
        text = "Neutron stars are the collapsed cores of massive stars, so dense that a teaspoon of their material would weigh billions of tons on Earth.",
    ),
    (
        id = "starship-development",
        title = "SpaceX Starship Development",
        category = "space",
        text = "Starship is designed as a fully reusable launch system intended for missions to the Moon, Mars, and beyond, using a stack of a booster and an upper-stage spacecraft.",
    ),
    (
        id = "cosmic-microwave-background",
        title = "Cosmic Microwave Background",
        category = "space",
        text = "The CMB is the leftover thermal radiation from roughly 380,000 years after the Big Bang, offering a snapshot of the early universe's structure.",
    ),
]

function validate_embeddings(embeddings, expected_count::Int)
    length(embeddings)==expected_count||throw(
        DimensionMismatch(
            "embedding service returned $(length(embeddings)) vectors for $(expected_count) texts",
        ),
    )
    expected_count>0||throw(ArgumentError("at least one text is required"))
    dim=length(first(embeddings))
    dim>0||throw(ArgumentError("embedding service returned an empty vector"))

    for embedding in embeddings
        length(embedding)==dim||throw(
            DimensionMismatch("embedding service returned inconsistent vector dimensions"),
        )
        all(isfinite, embedding)||throw(
            ArgumentError("embedding service returned a non-finite value"),
        )
    end

    return embeddings
end

"""Generate a batch of embeddings with Ollama's local `/api/embed` endpoint."""
function ollama_embeddings(
    texts::AbstractVector{<:AbstractString};
    model::AbstractString = DEFAULT_EMBEDDING_MODEL,
    url::AbstractString = DEFAULT_OLLAMA_URL,
)
    isempty(texts)&&throw(ArgumentError("at least one text is required"))
    request_body=JSON3.write((
        model = String(model),
        input = String[text for text in texts],
    ))
    response_body=IOBuffer()

    response=try
        Downloads.request(
            String(url);
            method = "POST",
            headers = ["Content-Type"=>"application/json"],
            input = IOBuffer(request_body),
            output = response_body,
        )
    catch error
        message=sprint(showerror, error)
        throw(
            ErrorException(
                "could not reach Ollama at $(url). Start Ollama and run `ollama pull $(model)`. Original error: $(message)",
            ),
        )
    end

    response.status==200||throw(
        ErrorException(
            "Ollama embedding request failed with HTTP status $(response.status)",
        ),
    )
    payload=JSON3.read(String(take!(response_body)))
    hasproperty(payload, :embeddings)||throw(
        ArgumentError("Ollama response is missing embeddings"),
    )
    embeddings=[Float32[value for value in embedding] for embedding in payload.embeddings]
    return validate_embeddings(embeddings, length(texts))
end

function search_documents(
    documents::AbstractVector,
    query::AbstractString;
    embedder::Function = ollama_embeddings,
    k::Int = 3,
)
    isempty(documents)&&throw(ArgumentError("at least one document is required"))
    1<=k<=length(documents)||throw(
        ArgumentError("k must be between one and the document count"),
    )
    texts=String[document.text for document in documents]
    embeddings=embedder(vcat(texts, [String(query)]))
    validate_embeddings(embeddings, length(texts)+1)
    document_vectors=reduce(hcat, embeddings[1:(end-1)])
    query_vector=last(embeddings)
    metadata=[
        (title = document.title, category = document.category, text = document.text) for
        document in documents
    ]
    ids=[document.id for document in documents]
    nlists=min(2, length(documents))

    return mktempdir() do path
        config=MaintenanceConfig(enabled = false)
        db=create_db(
            path;
            dim = size(document_vectors, 1),
            metric = :cosine,
            maintenance_config = config,
        )

        try
            insert!(db, document_vectors, metadata; ids = ids)
            build!(db; nlists = nlists, iterations = 8, seed = 42)
            save!(db)
        finally
            close(db)
        end

        loaded=load_db(path; maintenance_config = config, mmap_vectors = true)

        try
            return search(loaded, query_vector; k = k, nprobe = nlists, strategy = :ivf)
        finally
            close(loaded)
        end
    end
end

function main(args = ARGS)
    query=isempty(args) ? "How do scientists spot planets orbiting distant stars?" :
          join(args, " ")
    model=get(ENV, "SEN_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    url=get(ENV, "SEN_OLLAMA_URL", DEFAULT_OLLAMA_URL)
    embedder=texts->ollama_embeddings(texts; model = model, url = url)
    results=search_documents(SAMPLE_DOCUMENTS, query; embedder = embedder, k = 3)

    println("Query: $(query)")
    println("Top semantic matches:")

    for (result_number, result) in enumerate(results)
        println(
            "$(result_number). $(result.metadata.title) [score=$(round(result.score;digits=4))]",
        )
        println("   $(result.metadata.text)")
    end

    return results
end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    main()
end
