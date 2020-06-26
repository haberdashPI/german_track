module GermanTrack

using Tables, DrWatson, SignalOperators, WAV, Infiltrator, DataFrames,
    Printf, ProgressMeter, FileIO, EEGCoding, Query, Dates, Distributed,
    Unitful, DependentBootstrap, Distributions, LambdaFn, VegaLite, CSV,
    ProximalOperators, PlotAxes, AxisArrays, DataFramesMeta, Random, Statistics,
    JSON3, PyCall

using BSON: @save, @load
export CSV, @save, @load

include("analyses.jl")
include("util.jl")
include("features.jl")
include("classifier.jl")
include("stimuli.jl")
include("files.jl")
include("train_test.jl")

export processed_datadir, raw_datadir, stimulus_dir, raw_stim_dir, plotsdir

processed_datadir(args...) = joinpath(datadir(),"processed",args...)
raw_datadir(args...) = joinpath(datadir(),"raw",args...)
stimulus_dir() = processed_datadir("stimuli")
raw_stim_dir() = raw_datadir("stimuli")

# load and organize metadata about the stimuli
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

const numpy = PyNULL()

function __init__()
    cache_dir = joinpath(projectdir(),"_research","cache")
    EEGCoding.set_cache_dir!(cache_dir)
    copy!(numpy, pyimport_conda("numpy","numpy"))
    isdir(cache_dir) || mkdir(cache_dir)
end

end
