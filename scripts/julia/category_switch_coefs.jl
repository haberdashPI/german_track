# Setup
# =================================================================

using DrWatson
@quickactivate("german_track")
use_cache = true
seed = 11_18_2019
n_winlens = 6
n_folds = 10

using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures,
    Dates, Underscores, Random, Printf, ProgressMeter, VegaLite, FileIO,
    StatsBase, Bootstrap, BangBang, Transducers, PyCall, ScikitLearn, Flux,
    JSON3, JSONTables, Tables, Infiltrator, FileIO, BlackBoxOptim, RCall, Peaks, Formatting,
    Distributions

R"library(ggplot2)"
R"library(dplyr)"

DrWatson._wsave(file, data::Dict) = open(io -> JSON3.write(io, data), file, "w")

import GermanTrack: stim_info, speakers, directions, target_times, switch_times

# then, whatever choice we make, run an analysis  to evaluate
# the tradeoff of λ and % correct

wmeanish(x,w) = iszero(sum(w)) ? 0.0 : mean(coalesce.(x,one(eltype(x))/2),weights(w))

dir = joinpath(plotsdir(), string("results_", Date(now())))
isdir(dir) || mkdir(dir)

# is freq means always the same?

# Mean Frequency Bin Analysis
# =================================================================

classdf_file = joinpath(cache_dir("features"), savename("switch-freqmeans",
    (n_winlens = n_winlens,), "csv"))

spread(scale,npoints)   = x -> spread(x,scale,npoints)
spread(x,scale,npoints) = quantile.(Normal(x,scale/2),range(0.05,0.95,length=npoints))

if use_cache && isfile(classdf_file)
    classdf = CSV.read(classdf_file)
else
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    classdf_groups = @_ events |>
        filter(ishit(_, region = "target") ∈ ["hit", "reject"], __) |>
        groupby(__, [:sid, :condition])

    classdf = mapreduce(append!!, [:near, :far]) do class
        windowfn = class == :near ? windowswitch :
            windowbaseline(mindist = 0.5, minlength = 0.5, onempty = missing)
        result = compute_freqbins(subjects, classdf_groups, windowfn,
            [(len = winlen, start = 0) for winlen in spread(1, 0.5, n_winlens)])
        result[!,:switchclass] .= string(class)

        result
    end

    CSV.write(classdf_file, classdf)
end

# Find λ
# =================================================================

shuffled_sids = @_ unique(classdf.sid) |> shuffle!(stableRNG(2019_11_18, :lambda_folds), __)
λ_folds = folds(2, shuffled_sids)
classdf[!,:fold] = in.(classdf.sid, Ref(Set(λ_folds[1][1]))) .+ 1

lambdas = 10.0 .^ range(-2, 0, length=100)

function findclass((key, sdf))
    result = testclassifier(LassoPathClassifiers(lambdas),
        data = sdf, y = :switchclass, X = r"channel", crossval = :sid,
        n_folds = n_folds, seed = stablehash(:cond_switch, 2019_11_18),
        maxncoef = size(sdf[:,r"channel"], 2),
        irls_maxiter = 600, weight = :weight, on_model_exception = :throw)
    result[!, keys(key)] .= permutedims(collect(values(key)))
end
predictdf = @_ classdf |> groupby(__, [:condition, :fold, :winlen]) |> pairs |> collect |>
    foldxt(append!!, Map(findclass), __)

classmeans = @_ predictdf |>
    groupby(__, [:winlen, :sid, :λ, :fold, :nzcoef, :condition]) |>
    combine(__, [:correct, :weight] => wmeanish => :mean)

classmean_summary = @_ classmeans |>
    groupby(__, [:λ, :sid, :condition, :fold]) |>
    combine(__, :mean => mean => :mean, :nzcoef => maximum => :nzcoef)

@vlplot() + vcat(
    classmean_summary |> @vlplot(facet = {row = {field = :condition}}) + @vlplot(
        width = 600, height = 50,
        :line,
        color = {:condition, type = :nominal},
        x = {:λ, scale = {type = :log}},
        y = {:nzcoef, aggregate = :max, type = :quantitative, scale = {domain = [0, 20]}}
    ),
    (
        classmean_summary |> @vlplot(
            width = 600, height = 400,
            x = {:λ, scale = {type = :log}}) +
        @vlplot(:line,
            y = {:mean, aggregate = :mean, type = :quantitative,
                    scale = {domain = [0.7, 1]}},
            color = {field = :condition, type = :nominal},
        ) +
        @vlplot(:errorband,
            y = {:mean, aggregate = :ci, type = :quantitative,
                    scale = {domain = [0.7, 1]}},
            color = {field = :condition, type = :nominal},
        )
    )
)

# subtract null model

meandiff = @_ filter(_.λ == 1.0, classmean_summary) |>
    deletecols!(__, [:λ, :nzcoef]) |>
    rename!(__, :mean => :nullmean) |>
    innerjoin(__, classmean_summary, on = [:condition, :sid, :fold]) |>
    transform!(__, [:mean,:nullmean] => (-) => :meandiff)

grandmeandiff = @_ meandiff |>
    groupby(__, [:λ, :fold]) |>
    combine(__, :meandiff => mean => :meandiff) |>
    sort!(__, [:λ])

# pick the largest valued λ, with a non-negative peak for meandiff
function pickλ(df)
    peaks = @_ maxima(df.meandiff) |>
        filter(df.meandiff[_] > 0, __)
    maxλ = argmax(df[peaks,:λ])
    df[peaks[maxλ],[:λ]]
end
λs = @_ grandmeandiff |> groupby(__,:fold) |> combine(pickλ,__)
λs[!,:fold_text] .= string.("Fold: ",λs.fold)
λs[!,:yoff] = [0.26, 0.26]

@vlplot() + vcat(
    classmean_summary |> @vlplot(facet = {row = {field = :condition}}) + @vlplot(
        width = 600, height = 50,
        :line,
        color = {:condition, type = :nominal},
        x = {:λ, scale = {type = :log}},
        y = {:nzcoef, aggregate = :max, type = :quantitative, scale = {domain = [0, 20]}}
    ),
    (
        @vlplot() +
        (
            meandiff |> @vlplot(
                width = 600, height = 400,
                x = {:λ, scale = {type = :log}}) +
            @vlplot(:line,
                y = {:meandiff, aggregate = :mean, type = :quantitative},
                color = {field = :condition, type = :nominal},
            ) +
            @vlplot(:errorband,
                y = {:meandiff, aggregate = :ci, type = :quantitative},
                color = {field = :condition, type = :nominal},
            )
        ) +
        (
            @vlplot(data = λs) +
            @vlplot({:rule, strokeDash = [4, 4], size = 3}, x = :λ,
                color = {value = "green"}) +
            @vlplot({:text, align = :left, dy = -8, size =  12, angle = 90},
                text = :fold_text, x = :λ, y = :yoff)
        )
    )
)

final_λs = vcat((DataFrame(sid = sid, λ = λ, fold = fi)
    for (fi, (λ, λ_fold)) in enumerate(zip(λs.λ, λ_folds))
    for sid in λ_fold[2])...)

# select the needed labmdas
classdf = innerjoin(classdf, final_λs, on = [:sid, :fold])

# Use selected lambdas to plot accuracyes
# =================================================================

# (this is just a slice through the plot abovve)

λsid = groupby(final_λs, :sid)
meandiff_slice = @_ meandiff |> filter(_.λ == first(λsid[(sid = _.sid,)].λ), __)

title = "% Correct - % Correct of Null Model"
pl = meandiff_slice |>
    @vlplot(config = {legend = {disable = true}, scale = {barBandPaddingInner = 0.4}},
        width = 200,
        height = 200,
        x = {:condition, type = :nominal, axis = {labelAngle = 0}, title = ""},
        title = "Near/Far Switch Classification Accuracy") +
    @vlplot({:bar, binSpacing = 100},
        color = {:condition, type = :nominal},
        y = {:meandiff, aggregate = :mean, type = :quantitative, title = title}) +
    @vlplot({:errorbar, ticks = {size = 10, color = "black"}},
        x = {:condition, type = :nominal},
        y = {:meandiff, aggregate = :stderr, type = :quantitative, title = title}) +
    @vlplot({:point, filled = true, opacity = 0.25, xOffset = -5, size = 15},
        color = {value = "black"},
        x = {:condition, type = :nominal},
        y = {:meandiff, type = :quantitative, scale = {domain = [0, 0.35], title = title}})

pl |> save(joinpath(dir, "switch_class.svg"))
pl |> save(joinpath(dir, "switch_class.png"))
