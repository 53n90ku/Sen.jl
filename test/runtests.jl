using Test
using Sen

@testset "Sen" begin
    db = create_db("test-db";dim = 128, metric = :cosine)

    @test db.path == "test-db"
    @test db.dim ==128
    @test db.metric ==:cosine
end

include("test_bench.jl")
include("test_metrics.jl")
include("test_exact.jl")
include("test_filters.jl")
include("test_ivf.jl")
