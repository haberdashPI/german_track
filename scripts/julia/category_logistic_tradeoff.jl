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
        for len in 2.0 .^ range(-1, 1, length = 7),
            start in [0; 2.0 .^ range(-2, 2, length = 6)]]
    eeg_files = dfhit = @_ readdir(processed_datadir("eeg")) |>
        filter(occursin(r".h5$", _), __)
    subjects = Dict(
        sidfor(file) => load_subject(
            joinpath(processed_datadir("eeg"), file), stim_info,
            encoding = RawEncoding()
        ) for file in eeg_files)

    events = @_ mapreduce(_.events, append!!, values(subjects))
    classdf_groups = @_ events |>
        filter(_.target_present, __) |>
        filter(ishit(_, region = "target") == "hit", __) |>
        groupby(__, [:sid, :condition])

    progress = Progress(length(classdf_groups), desc = "Computing frequency bins...")
    classdf = @_ classdf_groups |>
        combine(function(sdf)
            # compute features in each window
            function findwindows(window)
                result = if use_absolute_features
                    compute_powerbin_features(subjects[sdf.sid[1]].eeg, sdf,
                        windowtarget, window)
                else
                    compute_powerdiff_features(subjects[sdf.sid[1]].eeg, sdf,
                        windowtarget, window)
                end
                result[!, :winstart] .= window.start
                result[!, :winlen] .= window.len
                result
            end
            x = foldxt(append!!, Map(findwindows), windows)
            next!(progress)
            x
        end, __)
    ProgressMeter.finish!(progress)
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
            weight = :weight, maxncoef = size(sdf[:,r"channel"],2), irls_maxiter = 200,
            debug_model_errors = false)

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
λs[!,:yoff] = [0.22, 0.20]

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

# Save best window length and  λ
# -----------------------------------------------------------------

allmeandiff = @_ filter(_.λ == 1.0, means) |>
    deletecols!(__, [:λ, :nzcoef]) |>
    rename(__, :mean => :nullmean) |>
    innerjoin(__, means, on = [:winlen, :winstart, :comparison, :sid, :fold]) |>
    transform!(__, [:mean,:nullmean] => (-) => :meandiff)

λfold = groupby(λs,:fold)

winlen_meandiff = @_ allmeandiff |>
    filter(_.λ == only(λfold[(fold = _.fold,)].λ),__) |>
    groupby(__, [:winlen, :comparison, :sid, :fold]) |>
    combine(__, :meandiff => mean => :meandiff)

winlen_meandiff |>
    @vlplot(facet = {column = {field = "fold", type = :ordinal}}) +
    (@vlplot(x = :winlen, color = :comparison) +
     @vlplot({:line, point = true},
        y = {:meandiff, type = :quantitative, aggregate = :mean}) +
    @vlplot({:errorband, point = true},
        y = {:meandiff, type = :quantitative, aggregate = :ci}))

winlen_byfold = @_ winlen_meandiff |>
    groupby(__, [:winlen, :fold, :sid]) |>
    combine(__, :meandiff => mean => :meandiff)

winlen_byfold |>
    @vlplot(facet = {column = {field = "fold", type = :ordinal}}) +
    (@vlplot(x = :winlen) +
     @vlplot({:line, point = true},
        y = {:meandiff, type = :quantitative, aggregate = :mean}) +
    @vlplot({:errorband, point = true},
        y = {:meandiff, type = :quantitative, aggregate = :ci}))

bestwinlens = @_ winlen_byfold |>
    groupby(__, :fold) |>
    combine(__, [:winlen, :meandiff] => ((len,x) -> len[argmax(x)]) => :winlen)

# TODO: automate and then cross validate λ selection
# NOTE: code below is all old, an unchanged from copied file (category_grid.jl)

# Find Best Window Length
# =================================================================

@static if !use_slurm && !test_optimization

    object_winlen_means = @_ object_classpredict |>
        groupby(__, [:winstart, :winlen, :salience_label, :target_time_label, :sid]) |>
        combine(__, [:correct, :weight] => wmeanish => :correct) |>
        groupby(__, [:winlen, :salience_label, :target_time_label]) |>
        combine(__, :correct => mean) |>
        insertcols!(__, :condition => "object")

    spatial_winlen_means = @_ spatial_classpredict |>
        groupby(__, [:winstart, :winlen, :salience_label, :target_time_label, :sid]) |>
        combine(__, [:correct, :weight] => wmeanish => :correct) |>
        groupby(__, [:winlen, :salience_label, :target_time_label]) |>
        combine(__, :correct => mean) |>
        insertcols!(__, :condition => "spatial")

    best_windows = @_ vcat(object_winlen_means, spatial_winlen_means) |>
        groupby(__, [:salience_label, :target_time_label]) |>
        combine(__, [:winlen, :correct_mean] =>
            ((len, val) -> len[argmax(val)]) => :winlen)

    best_windows_file = joinpath(paramdir, savename("best-windows",
        (absolute = use_absolute_features, classifier = classifier), "json"))

    @tagsave best_windows_file Dict(
        :data => JSONTables.ObjectTable(Tables.columns(best_windows)),
        :seed => seed
    ) safe = true
end
