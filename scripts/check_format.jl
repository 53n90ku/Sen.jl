using JuliaFormatter

root=normpath(joinpath(@__DIR__, ".."))
cd(root) do
    command=Cmd([
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        "*.jl",
    ])
    files=filter(isfile, filter(!isempty, split(chomp(read(command, String)), '\n')))
    unformatted=String[]

    for file in files
        format_file(file; overwrite = false)||push!(unformatted, file)
    end

    isempty(unformatted)||error("JuliaFormatter required for: $(join(unformatted, ", "))")
    println("JuliaFormatter check passed for $(length(files)) files")
end
