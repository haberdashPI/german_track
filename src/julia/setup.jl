using JSON, Revise, DataFrames, Printf, SampledSignals,
    ProgressMeter, FileIO, MATLAB, EEGCoding, CSVFiles, Statistics,
    Tables, DataKnots, Tables, Dates, Distributed, Unitful,
    DependentBootstrap
using BSON: @save, @load

# include(joinpath(projectdir(),"util","util.jl"))
# include(joinpath(projectdir(),"util","train_speakers.jl"))
includet(joinpath(srcdir(),"julia","util","util.jl"))
includet(joinpath(srcdir(),"julia","util","train_stimuli.jl"))

dates = JSON.parsefile(joinpath(projectdir(),"dateconfig.json"))
EEGCoding.set_cache_dir!(joinpath(projectdir(),"_research","cache"))
data_dir = joinpath(datadir(),"exp_pro","eeg",dates["data_dir"])
stimulus_dir = joinpath(datadir(),"exp_pro","stimuli")

config = JSON.parsefile(joinpath(datadir(),"exp_raw","eeg","config.json"))
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
