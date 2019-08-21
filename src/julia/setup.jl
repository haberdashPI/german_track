using JSON, Revise, DataFrames, Printf, SampledSignals,
    ProgressMeter, FileIO, MATLAB, EEGCoding, CSVFiles, Statistics,
    Tables, Query, Dates, Distributed, Unitful,
    DependentBootstrap, Distributions
using BSON: @save, @load

# include(joinpath(projectdir(),"util","util.jl"))
# include(joinpath(projectdir(),"util","train_speakers.jl"))
includet(joinpath(srcdir(),"julia","util","util.jl"))
includet(joinpath(srcdir(),"julia","util","stimuli.jl"))
includet(joinpath(srcdir(),"julia","util","train_stimuli.jl"))

dates = JSON.parsefile(joinpath(projectdir(),"dateconfig.json"))
EEGCoding.set_cache_dir!(joinpath(projectdir(),"_research","cache"))
data_dir = joinpath(datadir(),"exp_pro","eeg",dates["data_dir"])
raw_data_dir = joinpath(datadir(),"exp_raw","eeg")
stimulus_dir = joinpath(datadir(),"exp_pro","stimuli",dates["stim_data_dir"])
