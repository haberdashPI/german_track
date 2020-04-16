using Pkg
pkg"add DrWatson"

using DrWatson
quickactivate(@__DIR__,"german_track")
pkg"add https://github.com/JuliaLang/TOML.jl.git"
Pkg.instantiate()

using TOML

config = joinpath(projectdir(),"install.toml")
if !isfile(config)
    error("Could not find install.toml: see README.md")
elseif !isdir(datadir())
    name = TOML.parsefile(config)["data"]
    symlink(name,datadir())
    @info "The folder `data` now links to $name."
else
    @info "Directory `data` has already been created."
end
