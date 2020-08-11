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

# Display of model coefficients
# =================================================================

# Compare coefficients across folds
# -----------------------------------------------------------------

## the point here is not easy visualization, but just to determine
## how consistent coeffcieints are across the folds (if they are
## consistent, maybe we do use it to visualize??)

centerlen = @_ classdf.winlen |> unique |> sort! |> __[4]
centerstart = @_ classdf.winstart |> unique |> __[argmin(abs.(__ .- 0.0))]

classdf_atlen = @_ classdf |> filter(_.winlen == centerlen && _.winstart == centerstart, __)

classcomps_atlen = [
    "global-v-object"  => @_(classdf_atlen |> filter(_.condition in ["global", "object"],  __)),
    "global-v-spatial" => @_(classdf_atlen |> filter(_.condition in ["global", "spatial"], __)),
    "object-v-spatial" => @_(classdf_atlen |> filter(_.condition in ["object", "spatial"], __))
]

coefdf = mapreduce(append!!, classcomps_atlen) do (comp, data)
    function findclass((key, sdf))
        result = testclassifier(LassoClassifier(sdf.λ |> first),
            data = sdf, y = :condition, X = r"channel",
            crossval = :sid, n_folds = 10,
            seed = hash((:cond_coef_timeline,2019_11_18)),
            irls_maxiter = 100, include_model_coefs = true,
            weight = :weight, on_model_exception = :throw)

        result[!, keys(key)] .= permutedims(collect(values(key)))
        result[!, :comparison] .= comp

        result
    end

    groups = pairs(groupby(data, :fold))
    foldl(append!!, Map(findclass), collect(groups)[1:1])
end

coefnames_ = pushfirst!(propertynames(coefdf[:,r"channel"]), :C)

coefvals = @_ coefdf |>
    groupby(__, [:label_fold, :fold, :comparison]) |>
    combine(__, coefnames_ .=> (only ∘ unique) .=> coefnames_)

function parsecoef(coef)
    parsed = match(r"channel_([0-9]+)_([a-z]+)",string(coef))
    if !isnothing(parsed)
        chanstr, freqbin = parsed[1], parsed[2]
        chan = parse(Int,chanstr)
        chan, freqbin
    elseif string(coef) == "C"
        missing, missing
    else
        error("Unexpected coefficient $coef")
    end
end

coef_spread = @_ coefvals |>
    stack(__, coefnames_, [:label_fold, :fold, :comparison],
        variable_name = :coef) |>
    transform!(__, :coef => ByRow(x -> parsecoef(x)[1]) => :channel) |>
    transform!(__, :coef => ByRow(x -> parsecoef(x)[2]) => :freqbin)

minabs(x) = x[argmin(abs.(x))]

coef_spread_means = @_ coef_spread |>
    filter(!ismissing(_.channel), __) |>
    groupby(__, [:freqbin, :channel, :comparison]) |>
    combine(__, :value => median => :value,
                :value => length => :N,
                :value => (x -> quantile(x, 0.75)) => :valuehigh,
                :value => (x -> quantile(x, 0.25)) => :valuelow)

pl = coef_spread_means |>
    @vlplot(
        facet = {
            column = {field = :freqbin, type = :ordinal,
            sort = ["delta","theta","alpha","beta","gamma"]}}) +
    (@vlplot(x = :channel) +
     @vlplot(:rule,
        y = :valuelow,
        y2 = :valuehigh, color = :comparison) +
     @vlplot({:point, filled = true},
        y = :value,
        color = {
            field = :comparison,
            condition = {test = "datum.value == 0", value = "rgba(0,0,0,0)"}}))

pl |> save(joinpath(dir,"coef_spread_means.png"))

# Look at non-cross-validated coefficients
# -----------------------------------------------------------------

meanλ = classdf_atlen.λ |> unique |> mean
coefdf_noxval = mapreduce(append!!, classcomps_atlen) do (comp, data)
    function findclass((key, sdf))
        result = testclassifier(LassoClassifier(meanλ),
            data = sdf, y = :condition, X = r"channel",
            crossval = :sid, n_folds = 1,
            seed = hash((:cond_coef_timeline,2019_11_18)),
            maxncoef = size(sdf[:,r"channel"],2),
            irls_maxiter = 100, include_model_coefs = true,
            weight = :weight, on_model_exception = :throw)

        result[!, keys(key)] .= permutedims(collect(values(key)))
        result[!, :comparison] .= comp

        result
    end

    groups = pairs(groupby(data, :fold))
    foldl(append!!, Map(findclass), collect(groups)[1:1])
end

coefvals_noxval = @_ coefdf_noxval |>
    groupby(__, [:fold, :comparison]) |>
    combine(__, [:correct, :weight] => wmeanish => :correct,
                coefnames_ .=> (only ∘ unique) .=> coefnames_)

coef_spread_noxval = @_ coefvals_noxval |>
    stack(__, coefnames_, [:comparison],
        variable_name = :coef) |>
    transform!(__, :coef => ByRow(x -> parsecoef(x)[1]) => :channel) |>
    transform!(__, :coef => ByRow(x -> parsecoef(x)[2]) => :freqbin)

pl = coef_spread_noxval |>
    @vlplot(
        facet = {
            column = {field = :freqbin, type = :ordinal,
            sort = ["delta","theta","alpha","beta","gamma"]}}) +
    (@vlplot(x = :channel) +
     @vlplot(:rule,
        y = :value,
        y2 = {datum = 0}, color = :comparison) +
     @vlplot({:point, filled = true},
        y = :value,
        color = {
            field = :comparison,
            condition = {test = "datum.value == 0", value = "rgba(0,0,0,0)"}}))

pl |> save(joinpath(dir,"coef_spread_means_noxval.png"))

# TODO: Plot MCCA filtered spectrograms



