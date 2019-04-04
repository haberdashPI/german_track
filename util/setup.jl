using Pkg; Pkg.activate("..")
using JSON
using MATLAB
using Revise

base_dir = realpath(joinpath(@__DIR__,".."))
_, name = splitdir(base_dir)
if name != "german_track"
    @warn("Expected root directory to be named 'german_track'. Was '$name'.")
end

includet(joinpath(@__DIR__,"util","util.jl"))

analysis_dir = joinpath(base_dir,"analyses")
cache_dir = joinpath(base_dir,"analyses","cache")
data_dir = joinpath(base_dir,"data")
stimulus_dir = joinpath(base_dir,"stimuli")

config = JSON.parse(joinpath(base_dir,"config.json"))
match = false
default_i = 1
for i in 1:length(config)
    if config[i]["host"] == "default"
        default_i = i
    elseif startswith(gethostname(),config[i]["host"])
        raw_data_dir = config[i]["raw_data_dir"]
        match = true
    end
end
if !match
    raw_data_dir = config[default_i]["raw_data_dir"]
end
