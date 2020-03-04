module GermanTrack

using Tables, DrWatson, SignalOperators, WAV, Infiltrator, DataFrames,
    Printf, ProgressMeter, FileIO, EEGCoding, Query, Dates, Distributed,
    Unitful, DependentBootstrap, Distributions, LambdaFn, CSV,
    ProximalOperators, AxisArrays, DataFramesMeta, Random, Statistics,
    JSON3

using BSON: @save, @load
export CSV, @save, @load

include("analyses.jl")
include("util.jl")
include("stimuli.jl")
include("train_test.jl")

export data_dir, raw_data_dir, stimulus_dir, raw_stim_dir, cache_dir

const datefile = open(joinpath(projectdir(),"dateconfig.json"))
const dates = JSON3.read(datefile)
atexit(() -> close(datefile))

data_dir() = joinpath(datadir(),"exp_pro","eeg",dates.data_dir)
raw_data_dir() = joinpath(datadir(),"exp_raw","eeg")
stimulus_dir() = joinpath(datadir(),"exp_pro","stimuli",dates.stim_data_dir)
raw_stim_dir() = joinpath(datadir(),"exp_raw","stimuli")
cache_dir() = joinpath(datadir(),"_research","cache")

const stim_file = open(joinpath(stimulus_dir(), "config.json"))
const stim_info = JSON3.read(stim_file)
atexit(() -> close(stim_file))
const speakers = stim_info.test_block_cfg.trial_target_speakers
const directions = stim_info.test_block_cfg.trial_target_dir
const target_times = stim_info.test_block_cfg.target_times
const switch_times = map(times -> times ./ stim_info.fs,stim_info.test_block_cfg.switch_times)

function __init__()
    isdir(cache_dir()) || mkdir(cache_dir())
    modelcache = joinpath(cache_dir(),"models")
    isdir(modelcache) || mkdir(modelcache)
    EEGCoding.set_cache_dir!(modelcache)
end

end
