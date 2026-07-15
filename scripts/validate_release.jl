using Pkg
using TOML
using UUIDs

root=normpath(joinpath(@__DIR__,".."))
project_path=joinpath(root,"Project.toml")
project=TOML.parsefile(project_path)
run_private_tests=!("--skip-tests" in ARGS)

function require_release(condition::Bool,message::String)
    condition||error(message)
    return nothing
end

require_release(get(project,"name",nothing)=="Sen","project name must be Sen")
require_release(haskey(project,"uuid"),"project uuid is missing")
require_release(haskey(project,"version"),"project version is missing")
require_release(!isempty(get(project,"authors",String[])),"project authors are missing")
UUID(project["uuid"])
version=VersionNumber(project["version"])
require_release(version>=v"0.1.0","project version must be atleast 0.1.0")
require_release(haskey(project,"compat")&&haskey(project["compat"],"julia"),"julia compatibility is missing")
require_release(haskey(project,"targets")&&"Test" in get(project["targets"],"test",String[]),"test target is missing")
require_release(isfile(joinpath(root,"src","Sen.jl")),"package entrypoint is missing")
run_private_tests&&require_release(isfile(joinpath(root,"test","runtests.jl")),"private test entrypoint is missing")

for dependency in keys(get(project,"deps",Dict()))
    require_release(haskey(project["compat"],dependency),"compatibility is missing for $(dependency)")
end

gitignore=read(joinpath(root,".gitignore"),String)
require_release(any(line->strip(line)=="Manifest.toml",split(gitignore,'\n')),"Manifest.toml must be ignored for the package release")

Pkg.activate(root)
Pkg.instantiate()

using Sen

require_release(stable_api()===STABLE_API_V1,"stable api identity is invalid")

for name in stable_api()
    require_release(isdefined(Sen,name),"stable api symbol $(name) is missing")
    require_release(Base.isexported(Sen,name),"stable api symbol $(name) is not exported")
end

println("release metadata passed for Sen $(version)")

if run_private_tests
    Pkg.test()
end
