using JSON3
using Sockets
using Test

include("semantic_search.jl")

function mock_embedding(text::AbstractString)
    lowered=lowercase(text)
    embedding=Float32[
        count(
            term->occursin(term, lowered),
            ("voyager", "interstellar", "heliosphere", "radioisotope"),
        ),
        count(
            term->occursin(term, lowered),
            ("jwst", "infrared", "dust", "galax", "lagrange"),
        ),
        count(
            term->occursin(term, lowered),
            ("perseverance", "jezero", "microbial", "rock sample"),
        ),
        count(
            term->occursin(term, lowered),
            ("black hole", "event horizon", "shadow", "m87"),
        ),
        count(
            term->occursin(term, lowered),
            ("solar flare", "sun's surface", "satellite", "power grid"),
        ),
        count(
            term->occursin(term, lowered),
            (
                "exoplanet",
                "transit",
                "dimming",
                "planet",
                "orbit",
                "distant star",
                "spot planet",
            ),
        ),
        count(
            term->occursin(term, lowered),
            ("iss", "space station", "microgravity", "research lab"),
        ),
        count(
            term->occursin(term, lowered),
            ("neutron star", "collapsed core", "dense", "massive star"),
        ),
        count(
            term->occursin(term, lowered),
            ("starship", "reusable", "launch system", "booster"),
        ),
        count(
            term->occursin(term, lowered),
            ("cmb", "microwave background", "thermal radiation", "early universe"),
        ),
    ]
    iszero(sum(abs2, embedding))&&(embedding[end]=1.0f0)
    return embedding/sqrt(sum(abs2, embedding))
end

function read_mock_request(socket)
    readline(socket)
    content_length=0

    while true
        line=readline(socket)
        isempty(strip(line))&&break
        name, value=split(line, ":"; limit = 2)
        lowercase(strip(name))=="content-length"&&(content_length=parse(Int, strip(value)))
    end

    content_length>0||error("mock request has no body")
    return JSON3.read(String(read(socket, content_length)))
end

function start_mock_ollama()
    server=listen(ip"127.0.0.1", 0)
    port=Int(last(getsockname(server)))
    task=@async begin
        socket=accept(server)

        try
            request=read_mock_request(socket)
            texts=String[text for text in request.input]
            body=JSON3.write((
                model = String(request.model),
                embeddings = [mock_embedding(text) for text in texts],
            ))
            write(socket, "HTTP/1.1 200 OK\r\n")
            write(socket, "Content-Type: application/json\r\n")
            write(socket, "Content-Length: $(ncodeunits(body))\r\n")
            write(socket, "Connection: close\r\n\r\n")
            write(socket, body)
            flush(socket)
        finally
            close(socket)
            close(server)
        end
    end
    return (url = "http://127.0.0.1:$(port)/api/embed", task = task)
end

@testset "semantic search example" begin
    mock=start_mock_ollama()
    embeddings=ollama_embeddings(
        ["Voyager crossed interstellar space", "Exoplanets transit distant stars"];
        url = mock.url,
        model = "mock",
    )
    wait(mock.task)

    @test length(embeddings)==2
    @test length(first(embeddings))==10
    @test embeddings[1]!=embeddings[2]

    mock=start_mock_ollama()
    results=search_documents(
        SAMPLE_DOCUMENTS,
        "How do scientists spot planets orbiting distant stars?";
        embedder = texts->ollama_embeddings(texts; url = mock.url, model = "mock"),
        k = 3,
    )
    wait(mock.task)

    @test length(results)==3
    @test first(results).id=="exoplanet-detection"
    @test first(results).metadata.title=="Exoplanet Detection Methods"
end
