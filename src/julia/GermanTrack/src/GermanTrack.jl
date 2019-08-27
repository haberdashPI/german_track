module GermanTrack

using Reexport, SampledSignals, Statistics, Tables, DrWatson, JSON

@reexport using DataFrames, Printf, ProgressMeter, FileIO,
    MATLAB, EEGCoding, Query, Dates, Distributed, Unitful, DependentBootstrap,
    Distributions, LambdaFn, RCall, VegaLite, CSV, ProximalOperators
using BSON: @save, @load
export CSV, JSON, @save, @load


include("util.jl")
include("stimuli.jl")
include("train_stimuli.jl")

export data_dir, raw_data_dir, stimulus_dir, raw_stim_dir

dates(str) = JSON.parsefile(joinpath(projectdir(),"dateconfig.json"))[str]

data_dir() = joinpath(datadir(),"exp_pro","eeg",dates("data_dir"))
raw_data_dir() = joinpath(datadir(),"exp_raw","eeg")
stimulus_dir() = joinpath(datadir(),"exp_pro","stimuli",dates("stim_data_dir"))
raw_stim_dir() = joinpath(datadir(),"exp_raw","stimuli")

function __init__()
    EEGCoding.set_cache_dir!(joinpath(projectdir(),"_research","cache"))
end

end
