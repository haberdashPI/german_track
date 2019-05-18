using Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using JSON, Revise, DataFrames, Printf, SampledSignals,
    ProgressMeter, FileIO, MATLAB, EEGCoding, CSVFiles, Statistics
using BSON: @save, @load

base_dir = realpath(joinpath(@__DIR__,".."))
_, name = splitdir(base_dir)
if name != "german_track"
    @warn("Expected root directory to be named 'german_track'. Was '$name'.")
end

includet(joinpath(base_dir,"util","util.jl"))

analysis_dir = joinpath(base_dir,"analyses")
EEGCoding.set_cache_dir!(joinpath(base_dir,"analyses","cache"))
data_dir = joinpath(base_dir,"data")
stimulus_dir = joinpath(base_dir,"stimuli")

config = JSON.parsefile(joinpath(base_dir,"config.json"))
ismatch = false
default_i = 1
for i in 1:length(config)
    if config[i]["host"] == "default"
        default_i = i
    elseif startswith(gethostname(),config[i]["host"])
        raw_data_dir = config[i]["raw_data_dir"]
        ismatch = true
    end
end
if !ismatch
    raw_data_dir = config[default_i]["raw_data_dir"]
end
