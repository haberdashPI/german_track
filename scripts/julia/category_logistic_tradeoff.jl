# Setup
# =================================================================

using DrWatson
@quickactivate("german_track")
use_cache = true
seed = 072189
n_fun_evals = 5_000
test_optimization = false
num_local_procs = 1
num_cluster_procs = 16
use_absolute_features = true
use_slurm = gethostname() == "lcap.cluster"
classifier = :logistic_l1

using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures,
    Dates, Underscores, Random, Printf, ProgressMeter, VegaLite, FileIO,
    StatsBase, Bootstrap, BangBang, Transducers, PyCall, ScikitLearn, Flux,
    JSON3, JSONTables, Tables, Infiltrator, FileIO, BlackBoxOptim, RCall, Peaks

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

# is freq means always the same?

# Mean Frequency Bin Analysis
# =================================================================

isdir(processed_datadir("features")) || mkdir(processed_datadir("features"))
classdf_file = joinpath(processed_datadir("features"), savename("cond-freaqmeans",
    (absolute = use_absolute_features,), "csv"))

if use_cache && isfile(classdf_file)
    classdf = CSV.read(classdf_file)
else
    windows = [(len = len, start = start, before = -len)
        for len in 2.0 .^ range(-1, 1, length = 10),
            start in [0; 2.0 .^ range(-2, 2, length = 10)]]

    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")
    classdf_groups = @_ events |>
        filter(_.target_present, __) |>
        filter(ishit(_, region = "target") == "hit", __) |>
        groupby(__, [:sid, :condition])

    classdf = computefreqbins(subjects, classdf_groups, windowtarget, windows)
    CSV.write(classdf_file, classdf)
end

# Model evaluation
# =================================================================

λ_folds = folds(2, unique(classdf.sid))
classdf[!,:fold] = in.(classdf.sid, Ref(Set(λ_folds[1][1]))) .+ 1

classcomps = [
    "global-v-object"  => @_(classdf |> filter(_.condition in ["global", "object"],  __)),
    "global-v-spatial" => @_(classdf |> filter(_.condition in ["global", "spatial"], __)),
    "object-v-spatial" => @_(classdf |> filter(_.condition in ["object", "spatial"], __))
]

# what we want
# - across all lambda's tested: find accuracy for each fold
#   across all window lengths and starts
# do this for each classification group

lambdas = 10.0 .^ range(-2, 0, length=100)
resultdf = mapreduce(append!!, classcomps) do (comp, data)
    groups = pairs(groupby(data, [:winstart, :winlen, :fold]))
    function findclass((key,sdf))
        result = Empty(DataFrame)

        result = testclassifier(LassoPathClassifiers(lambdas), data = sdf, y = :condition,
            X = r"channel", crossval = :sid, n_folds = 10, seed = 2017_09_16,
            weight = :weight, maxncoef = size(sdf[:,r"channel"],2), irls_maxiter = 400,
            on_model_exception = :print)

        result[!, keys(key)] .= permutedims(collect(values(key)))
        result[!, :comparison] .= comp

        result
    end
    foldxt(append!!, Map(findclass), collect(groups))
end

# λ vs % classification tradeoff
# -----------------------------------------------------------------

means = @_ resultdf |>
    groupby(__,[:winlen, :winstart, :comparison, :λ, :nzcoef, :sid, :fold]) |>
    combine(__, [:correct, :weight] => wmeanish => :mean)
bestmeans = @_ means |>
    groupby(__, [:comparison, :λ, :nzcoef, :sid, :fold]) |>
    combine(__ , :mean => maximum => :mean) # |>

pl = @vlplot() +
    vcat(
        bestmeans |> @vlplot(
            width = 750, height = 100,
            :line,
            color = {field = :comparison, type = :nominal},
            x = {:λ, scale = {type = :log}},
            y = {:nzcoef, aggregate = :max, type = :quantitative}
        ),
        (
            bestmeans |> @vlplot(
                width = 750, height = 400,
                x = {:λ, scale = {type = :log}},
                color = {field = :comparison, type = :nominal},
            ) +
            @vlplot(
                :line,
                y = {:mean, aggregate = :mean, type = :quantitative, scale = {domain = [0.5, 1]}},
            ) +
            @vlplot(
                :errorband,
                y = {:mean, aggregate = :ci, type = :quantitative}
            )
        )
    )

pl |> save(joinpath(dir, "lambdas_by_condition.pdf"))
pl |> save(joinpath(dir, "lambdas_by_condition.html"))

meandiff = @_ filter(_.λ == 1.0, bestmeans) |>
    deletecols!(__, [:λ, :nzcoef]) |>
    rename!(__, :mean => :nullmean) |>
    innerjoin(__, bestmeans, on = [:comparison, :sid, :fold]) |>
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
λs[!,:yoff] = [0.24, 0.26]

pl = @vlplot() +
    vcat(
        meandiff |> @vlplot(
            :line, width = 750, height = 100,
            color = {field = :comparison, type = :nominal},
            x     = {:λ, scale = {type = :log, domain = [0.01, 0.35]},
                     title = "Regularization Parameter (λ)"},
            y     = {:nzcoef, aggregate = :max, type = :quantitative,
                     title = "# of non-zero coefficients (max)"}
        ),(
            @vlplot() +
            (
                meandiff |> @vlplot(
                    width = 750, height = 400,
                    x     = {:λ, scale = {type = :log, domain = [0.01, 0.35]},
                             title = "Regularization Parameter (λ)"},
                    color = {field = :comparison, type = :nominal}) +
                @vlplot(:errorband,
                    y = {:meandiff, aggregate = :ci,   type = :quantitative,
                         title = "% Correct - % Correct of Null Model (Intercept Only)"}) +
                @vlplot(:line,
                    y = {:meandiff, aggregate = :mean, type = :quantitative})
            ) +
            (
                @vlplot(data = {values = [{}]}, encoding = {y = {datum = 0}}) +
                @vlplot(mark = {type = :rule, strokeDash = [2, 2], size = 2})
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

pl |> save(joinpath(dir, "relative_lambdas_by_condition.pdf"))
pl |> save(joinpath(dir, "relative_lambdas_by_condition.html"))

# Save best window length and λ
# -----------------------------------------------------------------

final_λs = vcat((DataFrame(sid = sid, λ = λ, fold = fi)
    for (fi, (λ, λ_fold)) in enumerate(zip(λs.λ, λ_folds))
    for sid in first(λ_fold))...)

paramdir = processed_datadir("classifier_params")
λfile = joinpath(paramdir, "best-lambdas.json")
CSV.write(λfile, final_λs)
