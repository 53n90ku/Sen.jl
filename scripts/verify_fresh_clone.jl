using Pkg

const ROOT=normpath(joinpath(@__DIR__, ".."))

function run_julia(
    project::AbstractString,
    expression::AbstractString,
    depot::AbstractString,
)
    command=`$(Base.julia_cmd()) --startup-file=no --project=$(project) -e $(expression)`
    run(addenv(command, "JULIA_DEPOT_PATH"=>depot, "JULIA_PKG_PRECOMPILE_AUTO"=>"0"))
end

mktempdir() do temporary_root
    checkout=joinpath(temporary_root, "Sen.jl")
    depot=joinpath(temporary_root, "julia-depot")
    run(`git clone --quiet --no-local $(ROOT) $(checkout)`)

    run_julia(checkout, "using Pkg; Pkg.instantiate(); Pkg.test()", depot)
    run_julia(joinpath(checkout, "examples"), "using Pkg; Pkg.instantiate()", depot)

    example_test=joinpath(checkout, "examples", "test_semantic_search.jl")
    command=`$(Base.julia_cmd()) --startup-file=no --project=$(joinpath(checkout,"examples")) $(example_test)`
    run(addenv(command, "JULIA_DEPOT_PATH"=>depot, "JULIA_PKG_PRECOMPILE_AUTO"=>"0"))

    println("Fresh-clone verification passed with an isolated Julia depot.")
end
