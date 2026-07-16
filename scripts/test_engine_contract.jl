using Test

include("validate_engine_contract.jl")
using .EngineContractValidation

root=normpath(joinpath(@__DIR__, ".."))
report=EngineContractValidation.load_contract(root)

@testset "engine contract specification" begin
    @test report.contract_version==v"1.0.0"
    @test report.scope=="embedded_single_node_vector_search_engine"
    @test Set(gate.id for gate in report.gates)==EngineContractValidation.REQUIRED_GATE_IDS
    @test count(gate->gate.status=="pass", report.gates)==11
    @test isempty(EngineContractValidation.blocking_gates(report))
    @test EngineContractValidation.engine_ready(report)
    @test EngineContractValidation.main(["--quiet"])==0
    @test redirect_stderr(devnull) do
        EngineContractValidation.main(["--quiet", "--enforce"])
    end==0
    @test_throws ArgumentError EngineContractValidation.main(["--unknown"])
end
