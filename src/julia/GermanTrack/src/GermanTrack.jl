module GermanTrack

using Tables, DrWatson, SignalOperators, WAV, Infiltrator, DataFrames,
    Printf, ProgressMeter, FileIO, EEGCoding, Query, Dates, Distributed,
    Unitful, DependentBootstrap, Distributions, LambdaFn, VegaLite, CSV,
    ProximalOperators, PlotAxes, AxisArrays, DataFramesMeta, Random, Statistics,
    JSON3

using BSON: @save, @load
export CSV, @save, @load

include("analyses.jl")
include("util.jl")
include("stimuli.jl")
include("train_test.jl")

export data_dir, raw_data_dir, stimulus_dir, raw_stim_dir, plotsdir

const datefile = open(joinpath(projectdir(),"dateconfig.json"))
const dates = JSON3.read(datefile)
atexit(() -> close(datefile))

data_dir() = joinpath(datadir(),"exp_pro","eeg",dates.data_dir)
raw_data_dir() = joinpath(datadir(),"exp_raw","eeg")
stimulus_dir() = joinpath(datadir(),"exp_pro","stimuli",dates.stim_data_dir)
raw_stim_dir() = joinpath(datadir(),"exp_raw","stimuli")

# load and organize data about the stimuli
const stim_file = open(joinpath(stimulus_dir(), "config.json"))
const stim_info = JSON3.read(stim_file)
atexit(() -> close(stim_file))
const speakers = stim_info.test_block_cfg.trial_target_speakers
const directions = stim_info.test_block_cfg.trial_target_dir
const target_times = stim_info.test_block_cfg.target_times
const target_salience =
    CSV.read(joinpath(stimulus_dir(), "target_salience.csv")).salience |> Array
const switch_times = map(times -> times ./ stim_info.fs,stim_info.test_block_cfg.switch_times)

# define some useful categories for these stimuli
const salience_label = begin
    med = median(target_salience)
    ifelse.(target_salience .< med,"low","high")
end
const target_time_label = begin
    early = @_ DataFrame(
        time = target_times,
        switches = switch_times,
        row = 1:length(target_times)) |>
    map(sum(_1.time .> _1.switches) <= 2 ? "early" : "late",eachrow(__))
end

function __init__()
    cache_dir = joinpath(projectdir(),"_research","cache")
    EEGCoding.set_cache_dir!(cache_dir)
    isdir(cache_dir) || mkdir(cache_dir)
end

end
