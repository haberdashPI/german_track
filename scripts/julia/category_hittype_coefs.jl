# Setup
# =================================================================

n_winlens = 12
n_folds = 10

using DrWatson; @quickactivate("german_track")
using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures, Dates, Underscores,
    Printf, ProgressMeter, VegaLite, FileIO, StatsBase, BangBang, Transducers,
    Infiltrator, Peaks, Distributions, DSP, Random, CategoricalArrays, StatsModels,
    StatsFuns

dir = mkpath(plotsdir("hitmiss"))

# Find λ
# =================================================================

# Mean Frequency Bin Analysis
# -----------------------------------------------------------------

classdf_file = joinpath(cache_dir("features"), "hitmiss-freqmeans.csv")

if isfile(classdf_file)
    classdf = CSV.read(classdf_file)
else
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    classdf_groups = @_ events |>
        transform!(__, AsTable(:) => ByRow(x -> ishit(x, region = "target")) => :hittype) |>
        filter(_.hittype ∈ ["hit", "miss"], __) |>
        groupby(__, [:sid, :condition, :salience_label, :hittype])

    windows = [(len = len, start = start, before = -len)
        for len in 2.0 .^ range(-1, 1, length = 10),
            start in [0; 2.0 .^ range(-2, 2, length = 10)]]
    classdf = compute_freqbins(subjects, classdf_groups, windowtarget, windows)

    CSV.write(classdf_file, classdf)
end

# Compute classification accuracy
# -----------------------------------------------------------------

resultdf_file = joinpath(cache_dir("models"), "hitmiss-lambda-class.csv")

shuffled_sids = @_ unique(classdf.sid) |> shuffle!(stableRNG(2019_11_18, :lambda_folds,
    :salience, :hitmiss), __)
λ_folds = folds(2, shuffled_sids)
classdf[!,:fold] = in.(classdf.sid, Ref(Set(λ_folds[1][1]))) .+ 1

if isfile(resultdf_file) && mtime(resultdf_file) > mtime(classdf_file)
    resultdf = CSV.read(resultdf_file)
else
    lambdas = 10.0 .^ range(-2, 0, length=100)
    factors = [:fold, :winlen, :winstart, :condition, :salience_label]
    groups = groupby(classdf, factors)

    progress = Progress(length(groups))
    function findclass((key, sdf))
        result = testclassifier(LassoPathClassifiers(lambdas),
            data = sdf, y = :hittype, X = r"channel", crossval = :sid,
            n_folds = n_folds, seed = stablehash(:salience_classification, 2019_11_18),
            maxncoef = size(sdf[:,r"channel"], 2),
            irls_maxiter = 600, weight = :weight, on_model_exception = :throw)
        result[!, keys(key)] .= permutedims(collect(values(key)))
        next!(progress)

        result
    end

    resultdf = @_ groups |> pairs |> collect |>
        foldxt(append!!, Map(findclass), __)

    ProgressMeter.finish!(progress)
    CSV.write(resultdf_file, resultdf)

    alert("Completed salience/target-time classification!")
end

# λ selection
# -----------------------------------------------------------------

means = @_ resultdf |>
    groupby(__, [:condition, :λ, :nzcoef, :sid, :fold, :winstart, :winlen, :salience_label]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :mean)

bestmeans = @_ means |>
    groupby(__, [:condition, :λ, :nzcoef, :sid, :fold]) |>
    combine(__, :mean => maximum => :mean,
                :mean => logit ∘ shrinktowards(0.5, by = 0.01) ∘ maximum => :logitmean)

logitmeandiff = @_ filter(_.λ == 1.0, bestmeans) |>
    deletecols!(__, [:λ, :nzcoef, :mean]) |>
    rename!(__, :logitmean => :logitnullmean) |>
    innerjoin(__, bestmeans, on = [:condition, :sid, :fold]) |>
    transform!(__, [:logitmean,:logitnullmean] => (-) => :logitmeandiff)

grandlogitmeandiff = @_ logitmeandiff |>
    groupby(__, [:λ, :fold]) |>
    combine(__, :logitmeandiff => mean => :logitmeandiff) |>
    sort!(__, [:λ]) |>
    groupby(__, [:fold]) |>
    transform!(__, :logitmeandiff =>
        (x -> filtfilt(digitalfilter(Lowpass(0.3), Butterworth(5)), x)) => :logitmeandiff)

pl = grandlogitmeandiff |> @vlplot() +
    @vlplot(:line,
        config = {},
        color = {:fold, type = :nominal,
            legend = {orient = :none, legendX = 175, legendY = 0.5}},
        x     = {:λ, scale = {type = :log, domain = [0.01, 0.35]},
        title = "Regularization Parameter (λ)"},
        y     = {:logitmeandiff, aggregate = :mean, type = :quantitative,
                title = "Model - Null Model Accuracy (logit scale)"}) |>
    save(joinpath(dir, "grandmean.svg"))

# pick the largest valued λ, with a non-negative peak for meandiff
function pickλ(df)
    peaks = @_ maxima(df.logitmeandiff) |>
        filter(df.logitmeandiff[_] > 0.1, __)
    maxλ = argmax(df[peaks,:λ])
    df[peaks[maxλ],[:λ]]
end
λs = @_ grandlogitmeandiff |> groupby(__,:fold) |> combine(pickλ,__)

λs[!,:fold_text] .= string.("Fold: ",λs.fold)
λs[!,:yoff] = [0.1,0.15]

final_λs = vcat((DataFrame(sid = sid, λ = λ, fold = fi)
    for (fi, (λ, λ_fold)) in enumerate(zip(λs.λ, λ_folds))
    for sid in λ_fold[2])...)

pl = @vlplot() +
    vcat(
        logitmeandiff |> @vlplot(
        :line, width = 750, height = 100,
            color = {field = :condition, type = :nominal},
            x     = {:λ, scale = {type = :log, domain = [0.01, 0.35]},
                     title = "Regularization Parameter (λ)"},
            y     = {:nzcoef, aggregate = :max, type = :quantitative,
                     title = "# of non-zero coefficients (max)"}
        ),
        (
            @_ bestmeans |> DataFrames.transform(__, :mean => ByRow(x -> 100x) => :mean) |>
            @vlplot(
                width = 750, height = 200,
                x = {:λ, scale = {type = :log}},
                color = {field = :condition, type = :nominal},
            ) +
            @vlplot(
                :line,
                y = {:mean, aggregate = :mean, type = :quantitative,
                    title = "% Correct", scale = {domain = [50, 100]}},
            ) +
            @vlplot(
                :errorband,
                y = {:mean, aggregate = :ci, type = :quantitative}
            )
        ),
        (
            @vlplot() +
            (
                logitmeandiff |> @vlplot(
                    width = 750, height = 200,
                    x     = {:λ, scale = {type = :log},
                             title = "Regularization Parameter (λ)"},
                    color = {field = :condition, type = :nominal}) +
                @vlplot(:errorband,
                    y = {:logitmeandiff, aggregate = :ci,   type = :quantitative,
                         title = "Model - Null Model Accuracy (logit scale)"}) +
                @vlplot(:line,
                    y = {:logitmeandiff, aggregate = :mean, type = :quantitative})
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
    ) |> save(joinpath(dir, "lambdas.svg"))

# Plot timeline
# =================================================================

# Compute features
# -----------------------------------------------------------------

# NOTE: if this code ever diverges from the corresponding section in
# `category_salience_coefs.jl`, I need to rename the file here
classdf_timeline_file = joinpath(cache_dir("features"), "salience-freqmeans-timeline.csv")

if isfile(classdf_timeline_file)
    classdf_timeline = CSV.read(classdf_timeline_file)
else
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    classdf_timeline_groups = @_ events |>
        transform!(__, AsTable(:) => ByRow(x -> ishit(x, region = "target")) => :hittype) |>
        filter(_.hittype ∈ ["hit", "miss"], __) |>
        groupby(__, [:sid, :condition, :salience_label, :hittype])
    winbounds(start,k) = sid -> (start = start, len = winlen_bysid(sid) |>
        GermanTrack.spread(0.5,n_winlens,indices=k))

    windows = [winbounds(st,k) for st in range(0, 3, length = 64) for k in 1:n_winlens]
    classdf_timeline = compute_freqbins(subjects, classdf_timeline_groups, windowtarget,
        windows)

    CSV.write(classdf_timeline_file, classdf_timeline)
end

# Compute classification accuracy
# -----------------------------------------------------------------

λsid = groupby(final_λs, :sid)

resultdf_timeline_file = joinpath(cache_dir("models"), "hitmiss-timeline.csv")
classdf_timeline[!,:fold] = in.(classdf_timeline.sid, Ref(Set(λ_folds[1][1]))) .+ 1

if isfile(resultdf_timeline_file) && mtime(resultdf_timeline_file) > mtime(classdf_timeline_file)
    resultdf_timeline = CSV.read(resultdf_timeline_file)
else
    factors = [:fold, :winlen, :winstart, :condition, :salience_label]
    groups = groupby(classdf_timeline, factors)

    progress = Progress(length(groups))
    function findclass((key, sdf))
        λ = first(λsid[(sid = first(sdf.sid),)].λ)
        result = testclassifier(LassoPathClassifiers([1.0, λ]),
            data = sdf, y = :hittype, X = r"channel", crossval = :sid,
            n_folds = n_folds, seed = stablehash(:salience_classification, 2019_11_18),
            maxncoef = size(sdf[:,r"channel"], 2),
            irls_maxiter = 600, weight = :weight, on_model_exception = :throw)
        result[!, keys(key)] .= permutedims(collect(values(key)))
        next!(progress)

        result
    end

    resultdf_timeline = @_ groups |> pairs |> collect |>
        # foldl(append!!, Map(findclass), __)
        foldxt(append!!, Map(findclass), __)

    ProgressMeter.finish!(progress)
    CSV.write(resultdf_timeline_file, resultdf_timeline)

    alert("Completed salience timeline classification!")
end

# Plot results
# -----------------------------------------------------------------

classmeans = @_ resultdf_timeline |>
    groupby(__, [:winstart, :winlen, :sid, :λ, :fold, :condition, :salience_label]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :mean)

classmeans_sum = @_ classmeans |>
    groupby(__, [:winstart, :sid, :λ, :fold, :condition, :salience_label]) |>
    combine(__, :mean => mean => :mean)

nullmeans = @_ classmeans_sum |>
    filter(_.λ == 1.0, __) |>
    rename!(__, :mean => :nullmean) |>
    deletecols!(__, :λ)

classdiffs = let l = logit ∘ shrinktowards(0.5, by = 0.01)
    @_ classmeans_sum |>
        filter(_.λ != 1.0, __) |>
        innerjoin(__, nullmeans, on = [:winstart, :condition, :sid, :fold, :salience_label]) |>
        transform!(__, [:mean, :nullmean] => ByRow((x,y) -> l(x) - l(y)) => :logitmeandiff)
end

ytitle = "Model - Null Model Accuracy (logit scale)"
pl = classdiffs |>
    @vlplot(
        config = {legend = {disable = true}},
        title = "Hit/Miss Classification Accuracy",
        facet = {column = {field = :condition, type = :nominal}}) +
    (@vlplot(
        color = {field = :salience_label, type = :nominal},
    ) +
    # data lines
    @vlplot(:line,
        x = {:winstart, type = :quantitative, title = "Time (s) Relative to Target Onset"},
        y = {:logitmeandiff, aggregate = :mean, type = :quantitative, title = ytitle}) +
    # data errorbands
    @vlplot(:errorband,
        x = {:winstart, type = :quantitative, title = "Time (s) Relative to Target Onset"},
        y = {:logitmeandiff, aggregate = :ci, type = :quantitative, title = ytitle}) +
    # condition labels
    @vlplot({:text, align = :left, dx = 5},
        transform = [{filter = "datum.winstart > 2.2"}],
        x = {datum = 3.0},
        y = {:logitmeandiff, aggregate = :mean, type = :quantitative},
        text = :salience_label
    ) +
    # Basline (0 %) dotted line
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:rule, strokeDash = [4 4], size = 2},
            y = {datum = 0.0},
            color = {value = "black"})
    ) +
    # "Target Length" arrow annotation
    (
        @vlplot(data = {values = [
            {x = 0.05, y = 0.25, dir = 270},
            {x = 0.95, y = 0.25, dir = 90}]}) +
        @vlplot(mark = {:line, size = 1.5},
            x = {:x, type = :quantitative}, y = {:y, type = :quantitative},
            color = {value = "black"},
        ) +
        @vlplot(mark = {:point, shape = "triangle", opacity = 1.0, size = 10},
            x = {:x, type = :quantitative}, y = {:y, type = :quantitative},
            angle = {:dir, type = :quantitative, scale = {domain = [0, 360], range = [0, 360]}},
            color = {value = "black"}
        )
    ) +
    # "Target Length" text annotation
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:text, size = 11, baseline = "bottom", yOffset = -3},
            x = {datum = 0.5}, y = {datum = 0.25},
            text = {value = "Target Length"},
            color = {value = "black"}
        )
    ));
pl |> save(joinpath(dir, "hitmiss_timeline.svg"))

