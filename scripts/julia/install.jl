# add DrWatson to root environment
using Pkg
pkg"activate"
pkg"add DrWatson"

using DrWatson
quickactivate(@__DIR__,"german_track")
pkg"add https://github.com/JuliaLang/TOML.jl.git"
Pkg.instantiate()

using Conda
Conda.add("r-ggplot2",channel="r")
Conda.add("r-dplyr",channel="r")
Conda.add("r-cowplot",channel="biobuilds")

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
