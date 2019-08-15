using DrWatson, Pkg
quickactivate(@__DIR__,"german_track")
Pkg.instantiate()

using TOML

config = joinpath(projectdir(),"install.toml")
if !isfile(config)
    error("Could not find install.toml: see README.md")
elseif !isdir(datadir())
    name = TOML.parsefile(config)["data"]
    symlink(name,datadir()[1:end-1])
    @info "The folder `data` now links to $name."
else
    @info "Directory `data` has already been created."
end
