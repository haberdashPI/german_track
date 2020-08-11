# Setup
# =================================================================

using DrWatson
@quickactivate("german_track")
use_cache = true
seed = 072189
use_absolute_features = true
n_winlens = 6

using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures,
    Dates, Underscores, Random, Printf, ProgressMeter, VegaLite, FileIO,
    StatsBase, Bootstrap, BangBang, Transducers, PyCall, ScikitLearn, Flux,
    JSON3, JSONTables, Tables, Infiltrator, FileIO, BlackBoxOptim, RCall, Peaks,
    Distributions

R"library(ggplot2)"
R"library(dplyr)"

DrWatson._wsave(file, data::Dict) = open(io -> JSON3.write(io, data), file, "w")

# local only pac kages
using Formatting

import GermanTrack: stim_info, speakers, directions, target_times, switch_times

# then, whatever choice we make, run an analysis  to evaluate
# the tradeoff of λ and % correct

wmeanish(x,w) = iszero(sum(w)) ? 0.0 : mean(coalesce.(x,one(eltype(x))/2),weights(w))

dir = joinpath(plotsdir(), string("results_", Date(now())))
isdir(dir) || mkdir(dir)

best_λs = CSV.read(joinpath(processed_datadir("classifier_params"),"best-lambdas.json"))

# Mean Frequency Bin Analysis (Timeline)
# =================================================================

classdf_file = joinpath(cache_dir("features"), savename("cond-freaqmeans-timeline",
    (absolute = use_absolute_features, n_winlens = n_winlens), "csv"))

spread(scale,npoints)   = x -> spread(x,scale,npoints)
spread(x,scale,npoints) = quantile.(Normal(x,scale/2),range(0.05,0.95,length=npoints))

if use_cache && isfile(classdf_file)
    classdf = CSV.read(classdf_file)
else
    windows = [(len = len, start = start, before = -len)
        for len in spread(1, 0.5, n_winlens),
            start in range(-2.5, 2.5, length = 64)]

    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")
    classdf_groups = @_ events |>
        filter(_.target_present, __) |>
        filter(ishit(_, region = "target") == "hit", __) |>
        groupby(__, [:sid, :condition])

    classdf = compute_freqbins(subjects, classdf_groups, windowtarget, windows)
    CSV.write(classdf_file, classdf)
end
classdf = innerjoin(classdf, best_λs, on = [:sid])

# Timeline Plots
# =================================================================

classcomps = [
    "global-v-object"  => @_(classdf |> filter(_.condition in ["global", "object"],  __)),
    "global-v-spatial" => @_(classdf |> filter(_.condition in ["global", "spatial"], __)),
    "object-v-spatial" => @_(classdf |> filter(_.condition in ["object", "spatial"], __))
]

resultdf = mapreduce(append!!, classcomps) do (comp, data)
    function findclass((key, sdf))
        result = testclassifier(LassoPathClassifiers([1.0, sdf.λ |> first]),
            data = sdf, y = :condition, X = r"channel",
            crossval = :sid, n_folds = 10,
            seed = hash((:cond_coef_timeline,2019_11_18)),
            irls_maxiter = 100,
            weight = :weight, on_model_exception = :throw)

        result[!, keys(key)] .= permutedims(collect(values(key)))
        result[!, :comparison] .= comp

        result
    end

    groups = pairs(groupby(data, [:winstart, :winlen, :fold]))
    foldxt(append!!, Map(findclass), collect(groups))
end

means = @_ resultdf |>
    groupby(__, [:winlen, :winstart, :comparison, :sid, :fold, :λ]) |>
    combine(__, [:correct, :weight] => wmeanish => :mean)

winstart_means = @_ means |>
    groupby(__, [:winstart, :comparison, :sid, :fold, :λ]) |>
    combine(__, :mean => mean => :mean)

nullmeans = @_ winstart_means |>
    filter(_.λ == 1.0, __) |>
    deletecols!(__, :λ) |>
    rename!(__, :mean => :nullmean)

meandiff = @_ winstart_means |>
    filter(_.λ != 1.0, __) |>
    innerjoin(nullmeans, __, on = [:comparison, :sid, :fold, :winstart]) |>
    transform!(__, [:mean, :nullmean] => (-) => :meandiff)

pl = meandiff |>
    @vlplot(width = 600, height = 300,
        color = {field = :comparison, type = :nominal}, x = :winstart) +
    @vlplot(:line,  y = {:meandiff, aggregate = :mean, type = :quantitative}) +
    @vlplot(:errorband,  y = {:meandiff, aggregate = :ci, type = :quantitative})

pl |> save(joinpath(dir, "condition_timeline.pdf"))
pl |> save(joinpath(dir, "condition_timeline.html"))
