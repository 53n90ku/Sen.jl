module EngineContractValidation

using TOML

const CONTRACT_FORMAT_VERSION=1
const CONTRACT_STATES=Set(["accepted"])
const ALLOWED_STATUSES=Set(["pass","partial","fail"])
const REQUIRED_GATE_IDS=Set([
    "search_primitives",
    "durable_storage",
    "mutation_visibility",
    "valid_vector_admission",
    "writable_reopen",
    "atomic_mutations",
    "continuous_write_indexing",
    "bounded_mutation_search_cost",
    "recall_latency_contract",
    "crash_recovery_contract",
    "full_tests_in_ci",
])

struct ContractReport
    contract_version::VersionNumber
    scope::String
    claim::String
    gates::Vector{NamedTuple}
end

function require_contract(condition::Bool,message::AbstractString)
    condition||throw(ArgumentError(String(message)))
    return nothing
end

function required_string(table::AbstractDict,key::String,context::String)
    value=get(table,key,nothing)
    require_contract(value isa String&&!isempty(strip(value)),"$(context) $(key) must be a nonempty string")
    return String(value)
end

function repository_evidence_path(root::AbstractString,path::AbstractString)
    require_contract(!isabspath(path),"contract evidence paths must be relative")
    resolved=normpath(joinpath(root,path))
    relative=relpath(resolved,root)
    escapes=relative==".."||startswith(relative,"../")||startswith(relative,"..\\")
    require_contract(!escapes,"contract evidence path escapes the repository: $(path)")
    require_contract(ispath(resolved),"contract evidence does not exist: $(path)")
    return String(path)
end

function validate_policy(contract::AbstractDict)
    policy=get(contract,"policy",nothing)
    require_contract(policy isa AbstractDict,"contract policy table is missing")
    statuses=get(policy,"allowed_statuses",nothing)
    require_contract(statuses isa AbstractVector,"contract policy allowed_statuses must be an array")
    require_contract(Set(String.(statuses))==ALLOWED_STATUSES,"contract policy allowed_statuses must exactly match the validator")
    require_contract(get(policy,"engine_claim_requires_all_gates",nothing)===true,"engine claim must require every gate")
    required_string(policy,"validation_command","contract policy")
    required_string(policy,"enforcement_command","contract policy")
    return nothing
end

function validate_gate(root::AbstractString,gate::AbstractDict,index::Int)
    context="contract gate $(index)"
    id=required_string(gate,"id",context)
    title=required_string(gate,"title",context)
    requirement=required_string(gate,"requirement",context)
    status=required_string(gate,"status",context)
    require_contract(status in ALLOWED_STATUSES,"$(context) has invalid status $(repr(status))")
    stage=get(gate,"stage",nothing)
    require_contract(stage isa Integer&&stage>=0,"$(context) stage must be a nonnegative integer")
    evidence=get(gate,"evidence",nothing)
    require_contract(evidence isa AbstractVector&&!isempty(evidence),"$(context) must have evidence")
    evidence_paths=String[repository_evidence_path(root,required_string(Dict("path"=>path),"path","$(context) evidence")) for path in evidence]
    verification=required_string(gate,"verification",context)

    if status=="pass"
        require_contract(any(path->startswith(path,"test/")||startswith(path,"scripts/test_"),evidence_paths),"$(context) cannot pass without executable test evidence")
        require_contract(!startswith(lowercase(verification),"pending"),"$(context) cannot pass with pending verification")
    end

    return(id=id,title=title,requirement=requirement,status=status,stage=Int(stage),evidence=evidence_paths,verification=verification,)
end

function load_contract(root::AbstractString;path::AbstractString=joinpath(root,"engine_contract.toml"),)
    isfile(path)||throw(ArgumentError("engine contract does not exist: $(path)"))
    contract=TOML.parsefile(path)
    get(contract,"format_version",nothing)==CONTRACT_FORMAT_VERSION||throw(ArgumentError("unsupported engine contract format version"))
    state=required_string(contract,"contract_state","contract")
    require_contract(state in CONTRACT_STATES,"engine contract must be accepted")
    version=VersionNumber(required_string(contract,"contract_version","contract"))
    scope=required_string(contract,"scope","contract")
    claim=required_string(contract,"claim","contract")
    validate_policy(contract)
    raw_gates=get(contract,"gates",nothing)
    require_contract(raw_gates isa AbstractVector,"contract gates must be an array of tables")
    gates=NamedTuple[]

    for(index,gate) in enumerate(raw_gates)
        require_contract(gate isa AbstractDict,"contract gate $(index) must be a table")
        push!(gates,validate_gate(root,gate,index))
    end

    ids=String[gate.id for gate in gates]
    require_contract(length(ids)==length(Set(ids)),"contract gate IDs must be unique")
    require_contract(Set(ids)==REQUIRED_GATE_IDS,"contract gates do not match the frozen Stage 0 gate set")
    return ContractReport(version,scope,claim,gates)
end

engine_ready(report::ContractReport)=all(gate->gate.status=="pass",report.gates)
blocking_gates(report::ContractReport)=[gate for gate in report.gates if gate.status!="pass"]

function print_report(io::IO,report::ContractReport)
    println(io,"Sen engine contract $(report.contract_version)")
    println(io,"scope: $(report.scope)")
    println(io,"ready: $(engine_ready(report))")

    for gate in sort(report.gates;by=gate->(gate.stage,gate.id),)
        marker=gate.status=="pass" ? "PASS" : gate.status=="partial" ? "PARTIAL" : "FAIL"
        println(io,"[$(marker)] stage=$(gate.stage) $(gate.id): $(gate.title)")
    end

    return nothing
end

function main(args::Vector{String}=ARGS)
    known=Set(["--enforce","--quiet"])
    unknown=[arg for arg in args if !(arg in known)]
    isempty(unknown)||throw(ArgumentError("unknown arguments: $(join(unknown,", "))"))
    root=normpath(joinpath(@__DIR__,".."))
    report=load_contract(root)
    "--quiet" in args||print_report(stdout,report)

    if "--enforce" in args&&!engine_ready(report)
        blockers=join((gate.id for gate in blocking_gates(report)),", ")
        println(stderr,"engine contract is not satisfied; blocking gates: $(blockers)")
        return 1
    end

    return 0
end

end

if abspath(PROGRAM_FILE)==@__FILE__
    exit(EngineContractValidation.main())
end
