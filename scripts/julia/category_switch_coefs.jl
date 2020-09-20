# Setup
# =================================================================

using DrWatson
@quickactivate("german_track")
n_winlens = 6
n_folds = 10

using DrWatson; @quickactivate("german_track")
using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures, Dates, Underscores,
    Printf, ProgressMeter, VegaLite, FileIO, StatsBase, BangBang, Transducers,
    Infiltrator, Peaks, Distributions, DSP, Random, StatsFuns

dir = mkpath(plotsdir("category_switch"))

# is freq means always the same?

# Mean Frequency Bin Analysis
# =================================================================

classdf_file = joinpath(cache_dir("features"), savename("switch-freqmeans",
    (n_winlens = n_winlens,), "csv"))

if isfile(classdf_file)
    classdf = CSV.read(classdf_file)
else
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    classdf_groups = @_ events |>
        filter(ishit(_, region = "target") ∈ ["hit", "reject"], __) |>
        groupby(__, [:sid, :condition])

    classdf = mapreduce(append!!, [:near, :far]) do class
        windowfn = class == :near ? window_target_switch :
            windowbaseline(mindist = 0.5, minlength = 0.5, onempty = missing)
        result = compute_freqbins(subjects, classdf_groups, windowfn,
            [(len = winlen, start = 0) for winlen in GermanTrack.spread(1, 0.5, n_winlens)])
        result[!,:switchclass] .= string(class)

        result
    end

    CSV.write(classdf_file, classdf)
end

# Find λ
# =================================================================

# Classification accuracy
# -----------------------------------------------------------------
shuffled_sids = @_ unique(classdf.sid) |> shuffle!(stableRNG(2019_11_18, :lambda_folds), __)
λ_folds = folds(2, shuffled_sids)
classdf[!,:fold] = in.(classdf.sid, Ref(Set(λ_folds[1][1]))) .+ 1

lambdas = 10.0 .^ range(-2, 0, length=100)

groups = groupby(classdf, [:condition, :fold, :winlen])
progress = Progress(length(groups))
function findclass((key, sdf))
    result = testclassifier(LassoPathClassifiers(lambdas),
        data = sdf, y = :switchclass, X = r"channel", crossval = :sid,
        n_folds = n_folds, seed = stablehash(:cond_switch, 2019_11_18),
        maxncoef = size(sdf[:,r"channel"], 2),
        irls_maxiter = 600, weight = :weight, on_model_exception = :throw)
    result[!, keys(key)] .= permutedims(collect(values(key)))
    next!(progress)

    result
end
predictdf = @_ groups |> pairs |> collect |>
    foldxt(append!!, Map(findclass), __)

# λ selection
# -----------------------------------------------------------------

classmeans = @_ predictdf |>
    groupby(__, [:winlen, :sid, :λ, :fold, :nzcoef, :condition]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :mean)

classmean_summary = @_ classmeans |>
    groupby(__, [:λ, :sid, :condition, :fold]) |>
    combine(__,
        :mean => mean => :mean,
        :mean => (logit ∘ shrinktowards(0.5, by=0.01) ∘ mean) => :logitmean,
        :nzcoef => maximum => :nzcoef)

# subtract null model
meandiff = @_ filter(_.λ == 1.0, classmean_summary) |>
    deletecols!(__, [:λ, :nzcoef]) |>
    rename!(__, :logitmean => :logitnullmean) |>
    innerjoin(__, classmean_summary, on = [:condition, :sid, :fold]) |>
    transform!(__, [:logitmean,:logitnullmean] => (-) => :logitmeandiff)

grandmeandiff = @_ meandiff |>
    groupby(__, [:λ, :fold]) |>
    combine(__, :logitmeandiff => mean => :logitmeandiff) |>
    sort!(__, [:λ]) |>
    groupby(__, :fold) |>
    transform!(__, :logitmeandiff =>
        (x -> filtfilt(digitalfilter(Lowpass(0.3), Butterworth(5)), x)) => :logitmeandiff)

pl = grandmeandiff |> @vlplot() +
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
λs = @_ grandmeandiff |> groupby(__,:fold) |> combine(pickλ,__)
λs[!,:fold_text] .= string.("Fold: ",λs.fold)
λs[!,:yoff] = [0.26, 0.26]

final_λs = vcat((DataFrame(sid = sid, λ = λ, fold = fi)
    for (fi, (λ, λ_fold)) in enumerate(zip(λs.λ, λ_folds))
    for sid in λ_fold[2])...)

# select the needed labmdas
classdf = innerjoin(classdf, final_λs, on = [:sid, :fold])

@vlplot() + vcat(
    classmean_summary |> @vlplot() + @vlplot(
        width = 600, height = 100,
        :line,
        color = {:condition, type = :nominal},
        x = {:λ, scale = {type = :log}},
        y = {:nzcoef, aggregate = :max, type = :quantitative, scale = {domain = [0, 20]}}
    ),
    (
        @_ classmean_summary |>
            DataFrames.transform(__, :mean => ByRow(x -> 100x) => :mean) |>
        @vlplot(
            width = 600, height = 200,
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
            meandiff |> @vlplot(
                width = 600, height = 400,
                x = {:λ, scale = {type = :log}}) +
            @vlplot(:line,
                y = {:logitmeandiff, aggregate = :mean, type = :quantitative},
                color = {field = :condition, type = :nominal},
            ) +
            @vlplot(:errorband,
                y = {:logitmeandiff, aggregate = :ci, type = :quantitative},
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
) |> save(joinpath(dir, "switch_lambdas.svg"))

# Use selected lambdas to plot accuracyes
# =================================================================

# (this is just a slice through the plot abovve)

λsid = groupby(final_λs, :sid)
meandiff_slice = @_ meandiff |> filter(_.λ == first(λsid[(sid = _.sid,)].λ), __)

title = "Model - Null Model Accuracy (logit scale)"
pl = meandiff_slice |>
    @vlplot(config = {legend = {disable = true}, scale = {barBandPaddingInner = 0.4}},
        width = 200,
        height = 200,
        x = {:condition, type = :nominal, axis = {labelAngle = 0}, title = ""},
        title = "Near/Far Switch Classification Accuracy") +
    @vlplot({:bar, binSpacing = 100},
        color = {:condition, type = :nominal},
        y = {:logitmeandiff, aggregate = :mean, type = :quantitative, title = title}) +
    @vlplot({:errorbar, ticks = {size = 10, color = "black"}},
        x = {:condition, type = :nominal},
        y = {:logitmeandiff, aggregate = :stderr, type = :quantitative, title = title}) +
    @vlplot({:point, filled = true, opacity = 0.25, xOffset = -5, size = 15},
        color = {value = "black"},
        x = {:condition, type = :nominal},
        y = {:logitmeandiff, type = :quantitative, scale = {title = title}}) |>
    save(joinpath(dir, "switch_class.svg"))

# Break it down by target-time
# =================================================================

# Compute features
# -----------------------------------------------------------------

classdf_target_file = joinpath(cache_dir("features"), savename("switch-freqmeans-target",
    (n_winlens = n_winlens,), "csv"))

if isfile(classdf_target_file)
    classdf_target = CSV.read(classdf_target_file)
else
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    classdf_target_groups = @_ events |>
        filter(ishit(_, region = "target") ∈ ["hit", "reject"], __) |>
        groupby(__, [:sid, :condition, :target_time_label])

    classdf_target = mapreduce(append!!, [:near, :far]) do class
        windowfn = class == :near ? window_target_switch :
            windowbaseline(mindist = 0.5, minlength = 0.5, onempty = missing)
        result = compute_freqbins(subjects, classdf_target_groups, windowfn,
            [(len = winlen, start = 0) for winlen in GermanTrack.spread(1, 0.5, n_winlens)])
        result[!,:switchclass] .= string(class)

        result
    end

    CSV.write(classdf_target_file, classdf_target)
end

# compute classification accuracy
# -----------------------------------------------------------------

resultdf_target_file = joinpath(cache_dir("models"), "switch-freqmeans-target.csv")

shuffled_sids = @_ unique(classdf_target.sid) |> shuffle!(stableRNG(2019_11_18, :lambda_folds, :salience), __)
λ_folds = folds(2, shuffled_sids)
classdf_target[!,:fold] = in.(classdf_target.sid, Ref(Set(λ_folds[1][1]))) .+ 1

if isfile(resultdf_target_file) && mtime(resultdf_target_file) > mtime(classdf_target_file)
    resultdf_target = CSV.read(resultdf_target_file)
else
    lambdas = 10.0 .^ range(-2, 0, length=100)
    factors = [:fold, :winlen, :winstart, :condition, :target_time_label]
    groups = groupby(classdf_target, factors)

    progress = Progress(length(groups))
    function findclass((key, sdf))
        result = testclassifier(LassoPathClassifiers(lambdas),
            data = sdf, y = :switchclass, X = r"channel", crossval = :sid,
            n_folds = n_folds, seed = stablehash(:switch_classification, 2019_11_18),
            maxncoef = size(sdf[:,r"channel"], 2),
            irls_maxiter = 600, weight = :weight, on_model_exception = :throw)
        result[!, keys(key)] .= permutedims(collect(values(key)))
        next!(progress)

        result
    end

    resultdf_target = @_ groups |> pairs |> collect |>
        foldxt(append!!, Map(findclass), __)

    ProgressMeter.finish!(progress)
    CSV.write(resultdf_target_file, resultdf_target)
end


# plot early/late breakdown
# -----------------------------------------------------------------

λsid = groupby(final_λs, :sid)

means = @_ resultdf_target |>
    filter(_.λ ∈ (1.0, first(λsid[(sid = _.sid,)].λ)), __) |>
    groupby(__,[:condition, :sid, :fold, :λ, :winlen, :winstart, :target_time_label]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :mean) |>
    groupby(__, [:condition, :sid, :fold, :λ, :target_time_label]) |>
    combine(__, :mean => (logit ∘ shrinktowards(0.5, by=0.01) ∘ mean) => :logitmean)

nullmeans = @_ means |>
    filter(_.λ == 1.0, __) |>
    deletecols!(__, :λ) |>
    rename!(__, :logitmean => :logitnullmean)

meandiff = @_ means |>
    filter(_.λ != 1.0, __) |>
    innerjoin(nullmeans, __, on = [:condition, :sid, :fold, :target_time_label]) |>
    transform!(__, [:logitmean, :logitnullmean] => (-) => :logitmeandiff)

ytitle = "Model - Null Model Accuracy (logit scale)"
pl = meandiff |>
    @vlplot(
        facet = {column = {field = :condition, title = nothing}},
        title = ["Near/Far Classification ","Accuracy by Target Time"],
        config = {legend = {disable = true}}
    ) + (
        @vlplot(color = :condition, x = {:target_time_label, title = ["Target", "Time"]}) +
        @vlplot(:bar,
            y = {:logitmeandiff, aggregate = :mean, type = :quantitative, title = ytitle}
        ) +
        @vlplot(:errorbar,
            color = {value = "black"},
            y = {:logitmeandiff, aggregate = :ci, type = :quantitative, title = ytitle}
        )
    );

pl |> save(joinpath(dir, "switch_earlylate.svg"))
