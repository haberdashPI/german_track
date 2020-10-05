# Setup
# =================================================================

n_winlens = 12
n_folds = 10

using DrWatson; @quickactivate("german_track")
using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures, Dates, Underscores,
    Printf, ProgressMeter, VegaLite, FileIO, StatsBase, BangBang, Transducers,
    Infiltrator, Peaks, Distributions, DSP, Random, CategoricalArrays, StatsModels,
    StatsFuns

dir = mkpath(plotsdir("category_salience"))

# Find λ
# =================================================================

# Mean Frequency Bin Analysis
# -----------------------------------------------------------------

classdf_file = joinpath(cache_dir("features"), "salience-freqmeans.csv")

if isfile(classdf_file)
    classdf = CSV.read(classdf_file)
else
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    classdf_groups = @_ events |>
        filter(ishit(_, region = "target") ∈ ["hit"], __) |>
        groupby(__, [:sid, :condition, :salience_label])

    windows = [(len = len, start = start, before = -len)
        for len in 2.0 .^ range(-1, 1, length = 10),
            start in [0; 2.0 .^ range(-2, 2, length = 10)]]
    classdf = compute_freqbins(subjects, classdf_groups, windowtarget, windows)

    CSV.write(classdf_file, classdf)
end

# Compute classification accuracy
# -----------------------------------------------------------------

resultdf_file = joinpath(cache_dir("models"), "salience-target-time.csv")

shuffled_sids = @_ unique(classdf.sid) |> shuffle!(stableRNG(2019_11_18, :lambda_folds,
    :salience), __)
λ_folds = folds(2, shuffled_sids)
classdf[!,:fold] = in.(classdf.sid, Ref(Set(λ_folds[1][1]))) .+ 1

if isfile(resultdf_file) && mtime(resultdf_file) > mtime(classdf_file)
    resultdf = CSV.read(resultdf_file)
else
    lambdas = 10.0 .^ range(-2, 0, length=100)
    factors = [:fold, :winlen, :winstart, :condition]
    groups = groupby(classdf, factors)

    progress = Progress(length(groups))
    function findclass((key, sdf))
        result = testclassifier(LassoPathClassifiers(lambdas),
            data = sdf, y = :salience_label, X = r"channel", crossval = :sid,
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
    groupby(__, [:condition, :λ, :sid, :fold, :winstart, :winlen]) |>
    combine(__,
        :nzcoef => mean => :nzcoef,
        [:correct, :weight] => GermanTrack.wmean => :mean)

bestmeans = @_ means |>
    groupby(__, [:condition, :λ, :sid, :fold]) |>
    combine(__, :nzcoef => mean => :nzcoef,
                :mean => maximum => :mean,
                :mean => logit ∘ shrinktowards(0.5, by = 0.01) ∘ maximum => :logitmean)

# DEBUG: there should be three values per subject for a signle lambda value, but there are 100s

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
        (x -> filtfilt(digitalfilter(Lowpass(0.2), Butterworth(5)), x)) => :logitmeandiff)

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
    ) |> save(joinpath(dir, "salience_lambdas.svg"))

# Compute the best window length
# -----------------------------------------------------------------

best_λs = vcat((DataFrame(sid = sid, λ = λ, fold = fi)
    for (fi, (λ, λ_fold)) in enumerate(zip(λs.λ, λ_folds))
    for sid in λ_fold[2])...)
λsid = groupby(best_λs, :sid)

windowmeans = @_ resultdf |>
    filter(_.λ ∈ (1.0, first(λsid[(sid = _.sid,)].λ)), __) |>
    groupby(__,[:condition, :sid, :fold, :λ, :winlen, :winstart]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :mean) |>
    transform!(__, :mean => ByRow(logit ∘ shrinktowards(0.5, by = 0.01)) => :logitmean)

nullmeans = @_ windowmeans |>
    filter(_.λ == 1.0, __) |>
    deletecols!(__, [:λ, :mean]) |>
    rename!(__, :logitmean => :logitnullmean)

windowdiff = @_ windowmeans |>
    filter(_.λ != 1.0, __) |>
    innerjoin(nullmeans, __, on = [:condition, :sid, :fold, :winlen, :winstart]) |>
    transform!(__, [:logitmean, :logitnullmean] => (-) => :logitmeandiff)

windavg = @_ windowdiff |> groupby(__, [:condition, :fold, :winlen, :winstart]) |>
    combine(__, :logitmeandiff => mean => :logitmeandiff) |>
    groupby(__, [:fold, :winlen]) |>
    combine(__, :logitmeandiff => maximum => :logitmeandiff)

bestlens = @_ windavg |> groupby(__, [:fold]) |>
    combine(__, [:logitmeandiff, :winlen] =>
        ((m,l) -> l[argmax(m)]) => :winlen,
        :logitmeandiff => maximum => :logitmeandiff)

bestlen_bysid = @_ bestlens |>
    groupby(__, [:fold, :winlen, :logitmeandiff]) |>
    combine(__, :fold => (f -> λ_folds[f |> first][2]) => :sid) |>
    groupby(__, :sid)
    winlen_bysid(sid) = bestlen_bysid[(sid = sid,)].winlen |> first

pl = windowdiff |>
    @vlplot(:rect,
        config =  {view = {stroke = :transparent}},
        column = :condition,
        # row = :fold,
        y = {:winlen, type = :ordinal, axis = {format = ".2f"}, sort = :descending,
            title = "Length (s)"},
        x = {:winstart, type = :ordinal, axis = {format = ".2f"}, title = "Start (s)"},
        color = {:logitmeandiff, aggregate = :mean, type = :quantitative,
            scale = {scheme = "redblue", domainMid = 0}}) |>
    save(joinpath(dir, "salience_windows.svg"))

# Plot timeline
# =================================================================

# Compute frequency bins
# -----------------------------------------------------------------

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

resultdf_timeline_file = joinpath(cache_dir("models"), "salience-timeline.csv")
classdf_timeline[!,:fold] = in.(classdf_timeline.sid, Ref(Set(λ_folds[1][1]))) .+ 1

if isfile(resultdf_timeline_file) && mtime(resultdf_timeline_file) > mtime(classdf_timeline_file)
    resultdf_timeline = CSV.read(resultdf_timeline_file)
else
    factors = [:fold, :winlen, :winstart, :condition, :hittype]
    groups = groupby(classdf_timeline, factors)

    progress = Progress(length(groups))
    function findclass((key, sdf))
        λ = first(λsid[(sid = first(sdf.sid),)].λ)
        result = testclassifier(LassoPathClassifiers([1.0, λ]),
            data = sdf, y = :salience_label, X = r"channel", crossval = :sid,
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

# Display classification timeline
# -----------------------------------------------------------------

classmeans = @_ resultdf_timeline |>
    groupby(__, [:winstart, :winlen, :sid, :λ, :fold, :condition, :hittype]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :mean)

classmeans_sum = @_ classmeans |>
    groupby(__, [:winstart, :sid, :λ, :fold, :condition, :hittype]) |>
    combine(__, :mean => mean => :mean)

nullmeans = @_ classmeans_sum |>
    filter(_.λ == 1.0, __) |>
    rename!(__, :mean => :nullmean) |>
    deletecols!(__, :λ)

nullmean, classdiffs = let l = logit ∘ shrinktowards(0.5, by = 0.01), C = mean(l.(nullmeans.nullmean))
    100logistic(C),
    @_ classmeans_sum |>
        innerjoin(__, nullmeans, on = [:winstart, :condition, :sid, :fold, :hittype]) |>
        filter(_.λ != 1.0, __) |>
        transform!(__, [:mean, :nullmean] => ByRow((x,y) -> l(x) - l(y)) => :logitmeandiff)
end

annotate = @_ map(abs(_ - 3.0), classdiffs.winstart) |> classdiffs.winstart[argmin(__)]
ytitle = "Full - Null Model (logit scale)"
target_len_y = 0.1
pl = classdiffs |>
    @vlplot(
        config = {legend = {disable = true}},
        title = "Salience Classification Accuracy",
        facet = {column = {field = :hittype, type = :nominal}}) +
    (@vlplot(
        color = {field = :condition, type = :nominal},
    ) +
    # data lines
    @vlplot(:line,
        x = {:winstart, type = :quantitative, title = "Time (s) Relative to Target Onset"},
        y = {:logitmeandiff, aggregate = :mean, type = :quantitative, title = ytitle,
            #= scale = {domain = [50,67]} =#}) +
    # data errorbands
    @vlplot(:errorband,
        x = {:winstart, type = :quantitative, title = "Time (s) Relative to Target Onset"},
        y = {:logitmeandiff, aggregate = :ci, type = :quantitative, title = ytitle}) +
    # condition labels
    @vlplot({:text, align = :left, dx = 5},
        transform = [{filter = "datum.winstart > 2.2"}],
        x = {datum = 3.0},
        y = {:logitmeandiff, aggregate = :mean, type = :quantitative},
        text = :condition
    ) +
    # Basline (0 %) dotted line
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:rule, strokeDash = [4 4], size = 2},
            y = {datum = 0.0},
            color = {value = "black"})
    ) +
    # "Null Model" text annotation
    # (
    #     @vlplot(data = {values = [{}]}) +
    #     @vlplot(mark = {:text, size = 11, baseline = "line-top", dy = 4},
    #         x = {datum = 2.3}, y = {datum = nullmean},
    #         text = {value = ["Mean Accuracy", "of Null Model"]},
    #         color = {value = "black"}
    #     )
    # ) +
    # "Target Length" arrow annotation
    (
        @vlplot(data = {values = [
            {x = 0.05, y = target_len_y, dir = 270},
            {x = 0.95, y = target_len_y, dir = 90}]}) +
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
            x = {datum = 0.5}, y = {datum = target_len_y},
            text = {value = "Target Length"},
            color = {value = "black"}
        )
    ));
pl |> save(joinpath(dir, "salience_timeline.svg"))

hitvmiss = @_ classdiffs |>
    unstack(__, [:winstart, :sid, :condition], :hittype, :logitmeandiff) |>
    transform!(__, [:hit, :miss] => (-) => :hitvmiss)

annotate = @_ map(abs(_ - 3.0), classdiffs.winstart) |> classdiffs.winstart[argmin(__)]
ytitle = "Hit - Miss (logit scale)"
pl = hitvmiss |>
    @vlplot(
        config = {legend = {disable = true}},
        title = "Low/High Salience Classification Accuracy",
        # facet = {column = {field = :hittype, type = :nominal}}
    ) +
    (@vlplot(
        color = {field = :condition, type = :nominal},
    ) +
    # data lines
    @vlplot(:line,
        x = {:winstart, type = :quantitative, title = "Time (s) Relative to Target Onset"},
        y = {:hitvmiss, aggregate = :mean, type = :quantitative, title = ytitle}) +
    # data errorbands
    @vlplot(:errorband,
        x = {:winstart, type = :quantitative, title = "Time (s) Relative to Target Onset"},
        y = {:hitvmiss, aggregate = :ci, type = :quantitative, title = ytitle}) +
    # condition labels
    @vlplot({:text, align = :left, dx = 5},
        transform = [{filter = "datum.winstart > 2.2"}],
        x = {datum = 3.0},
        y = {:hitvmiss, aggregate = :mean, type = :quantitative},
        text = :condition
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
pl |> save(joinpath(dir, "salience_timeline_hitvmiss.svg"))

# Frontal vs. Central Salience Timeline
# =================================================================

# Compute frequency bins
# -----------------------------------------------------------------

classdf_chgroup_file = joinpath(cache_dir("features"), "salience-freqmeans-timeline-chgroups.csv")

if isfile(classdf_chgroup_file)
    classdf_chgroup = CSV.read(classdf_chgroup_file)
else
    function chgroup_freqmeans(group)
        subjects, events = load_all_subjects(processed_datadir("eeg", group), "h5")

        classdf_chgroup_groups = @_ events |>
            transform!(__, AsTable(:) => ByRow(x -> ishit(x, region = "target")) => :hittype) |>
            filter(_.hittype ∈ ["hit", "miss"], __) |>
            groupby(__, [:sid, :condition, :salience_label, :hittype])

        winbounds(start,k) = sid -> (start = start, len = winlen_bysid(sid) |>
            GermanTrack.spread(0.5,n_winlens,indices=k))

        windows = [winbounds(st,k) for st in range(0, 3, length = 64) for k in 1:n_winlens]
        result = compute_freqbins(subjects, classdf_chgroup_groups, windowtarget,
            windows)
        result[!, :chgroup] .= group

        result
    end
    classdf_chgroup = foldl(append!!, Map(chgroup_freqmeans), ["frontal", "central", "mixed"])

    CSV.write(classdf_chgroup_file, classdf_chgroup)
end

# Compute classification accuracy
# -----------------------------------------------------------------

λsid = groupby(final_λs, :sid)

resultdf_chgroup_file = joinpath(cache_dir("models"), "salience-chgroup.csv")
classdf_chgroup[!,:fold] = in.(classdf_chgroup.sid, Ref(Set(λ_folds[1][1]))) .+ 1

if isfile(resultdf_chgroup_file) && mtime(resultdf_chgroup_file) > mtime(classdf_chgroup_file)
    resultdf_chgroup = CSV.read(resultdf_chgroup_file)
else
    factors = [:fold, :winlen, :winstart, :condition, :hittype, :chgroup]
    groups = groupby(classdf_chgroup, factors)

    progress = Progress(length(groups))
    function findclass((key, sdf))
        λ = first(λsid[(sid = first(sdf.sid),)].λ)
        result = testclassifier(LassoPathClassifiers([1.0, λ]),
            data = sdf, y = :salience_label, X = r"channel", crossval = :sid,
            n_folds = n_folds, seed = stablehash(:salience_classification, :chgroup, 2019_11_18),
            maxncoef = size(sdf[:,r"channel"], 2),
            irls_maxiter = 600, weight = :weight, on_model_exception = :throw)
        result[!, keys(key)] .= permutedims(collect(values(key)))
        next!(progress)

        result
    end

    resultdf_chgroup = @_ groups |> pairs |> collect |>
        # foldl(append!!, Map(findclass), __)
        foldxt(append!!, Map(findclass), __)

    ProgressMeter.finish!(progress)
    CSV.write(resultdf_chgroup_file, resultdf_chgroup)

    alert("Completed salience timeline classification!")
end

# Timeline x Channel group
# -----------------------------------------------------------------

# Display classification timeline
# -----------------------------------------------------------------

classmeans = @_ resultdf_chgroup |>
    groupby(__, [:winstart, :winlen, :sid, :λ, :fold, :condition, :hittype, :chgroup]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :mean)

classmeans_sum = @_ classmeans |>
    groupby(__, [:winstart, :sid, :λ, :fold, :condition, :hittype, :chgroup]) |>
    combine(__, :mean => mean => :mean)

nullmeans = @_ classmeans_sum |>
    filter(_.λ == 1.0, __) |>
    rename!(__, :mean => :nullmean) |>
    deletecols!(__, :λ)

nullmean, classdiffs = let l = logit ∘ shrinktowards(0.5, by = 0.01), C = mean(l.(nullmeans.nullmean))
    100logistic(C),
    @_ classmeans_sum |>
        innerjoin(__, nullmeans, on = [:winstart, :condition, :sid, :fold, :hittype, :chgroup]) |>
        filter(_.λ != 1.0, __) |>
        transform!(__, [:mean, :nullmean] => ByRow((x,y) -> l(x) - l(y)) => :logitmeandiff)
end

annotate = @_ map(abs(_ - 3.0), classdiffs.winstart) |> classdiffs.winstart[argmin(__)]
ytitle = "Full - Null Model (logit scale)"
target_len_y = 1.0
pl = classdiffs |>
    @vlplot(
        config = {legend = {disable = true}},
        title = {text = "Salience Classification Accuracy", subtitle = "Object Condition"},
        facet = {column = {field = :hittype, type = :nominal}}) +
    (@vlplot(
        color = {field = :chgroup, type = :nominal, scale = {scheme = "set2"}},
    ) +
    # data lines
    @vlplot(:line,
        x = {:winstart, type = :quantitative, title = "Time (s) Relative to Target Onset"},
        y = {:logitmeandiff, aggregate = :mean, type = :quantitative, title = ytitle,
            #= scale = {domain = [50,67]} =#}) +
    # data errorbands
    @vlplot(:errorband,
        x = {:winstart, type = :quantitative, title = "Time (s) Relative to Target Onset"},
        y = {:logitmeandiff, aggregate = :ci, type = :quantitative, title = ytitle}) +
    # condition labels
    @vlplot({:text, align = :left, dx = 5},
        transform = [{filter = "datum.winstart > 2.2"}],
        x = {datum = 3.0},
        y = {:logitmeandiff, aggregate = :mean, type = :quantitative},
        text = :chgroup
    ) +
    # Basline (0 %) dotted line
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:rule, strokeDash = [4 4], size = 2},
            y = {datum = 0.0},
            color = {value = "black"})
    ) +
    # "Null Model" text annotation
    # (
    #     @vlplot(data = {values = [{}]}) +
    #     @vlplot(mark = {:text, size = 11, baseline = "line-top", dy = 4},
    #         x = {datum = 2.3}, y = {datum = nullmean},
    #         text = {value = ["Mean Accuracy", "of Null Model"]},
    #         color = {value = "black"}
    #     )
    # ) +
    # "Target Length" arrow annotation
    (
        @vlplot(data = {values = [
            {x = 0.05, y = target_len_y, dir = 270},
            {x = 0.95, y = target_len_y, dir = 90}]}) +
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
            x = {datum = 0.5}, y = {datum = target_len_y},
            text = {value = "Target Length"},
            color = {value = "black"}
        )
    ));
pl |> save(joinpath(dir, "salience_chgroup.svg"))

# Early/late targets
# =================================================================

# Mean Frequency Bin Analysis
# -----------------------------------------------------------------

classdf_earlylate_file = joinpath(cache_dir("features"), "salience-freqmeans-earlylate-timeline.csv")

if isfile(classdf_earlylate_file)
    classdf_earlylate = CSV.read(classdf_earlylate_file)
else
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    classdf_earlylate_groups = @_ events |>
        filter(ishit(_, region = "target") ∈ ["hit"], __) |>
        groupby(__, [:sid, :condition, :salience_label, :target_time_label])

    windows = [(len = len, start = start, before = -len)
        for len in 2.0 .^ range(-1, 1, length = 10),
            start in range(0, 2.5, length = 12)]
    classdf_earlylate = compute_freqbins(subjects, classdf_earlylate_groups, windowtarget,
        windows)

    CSV.write(classdf_earlylate_file, classdf_earlylate)
end

# Compute classification accuracy
# -----------------------------------------------------------------

λsid = groupby(final_λs, :sid)

resultdf_earlylate_file = joinpath(cache_dir("models"), "salience-earlylate-timeline.csv")
classdf_earlylate[!,:fold] = in.(classdf_earlylate.sid, Ref(Set(λ_folds[1][1]))) .+ 1

if isfile(resultdf_earlylate_file) && mtime(resultdf_earlylate_file) > mtime(classdf_earlylate_file)
    resultdf_earlylate = CSV.read(resultdf_earlylate_file)
else
    factors = [:fold, :winlen, :winstart, :condition, :target_time_label]
    groups = groupby(classdf_earlylate, factors)

    progress = Progress(length(groups))
    function findclass((key, sdf))
        λ = first(λsid[(sid = first(sdf.sid),)].λ)
        result = testclassifier(LassoPathClassifiers([1.0, λ]),
            data               = sdf,
            y                  = :salience_label,
            X                  = r"channel",
            crossval           = :sid,
            n_folds            = n_folds,
            seed               = stablehash(:salience_classification,
                                            :target_time, 2019_11_18),
            maxncoef           = size(sdf[:,r"channel"], 2),
            irls_maxiter       = 600,
            weight             = :weight,
            on_model_exception = :throw,
        )
        result[!, keys(key)] .= permutedims(collect(values(key)))
        next!(progress)

        result
    end

    resultdf_earlylate = @_ groups |> pairs |> collect |>
        foldxt(append!!, Map(findclass), __)

    ProgressMeter.finish!(progress)
    CSV.write(resultdf_earlylate_file, resultdf_earlylate)
end

# Plot salience by early/late targets
# -----------------------------------------------------------------

classmeans = @_ resultdf_earlylate |>
    groupby(__, [:winstart, :winlen, :sid, :λ, :fold, :condition, :target_time_label]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :mean)

classmeans_sum = @_ classmeans |>
    groupby(__, [:sid, :λ, :fold, :condition, :target_time_label]) |>
    combine(__, :mean => mean => :mean)

nullmeans = @_ classmeans_sum |>
    filter(_.λ == 1.0, __) |>
    rename!(__, :mean => :nullmean) |>
    deletecols!(__, :λ)

classdiffs = let l = logit ∘ shrinktowards(0.5, by = 0.01)
    @_ classmeans_sum |>
        innerjoin(__, nullmeans, on = [:condition, :sid, :fold, :target_time_label]) |>
        transform!(__, [:mean, :nullmean] => ByRow((x,y) -> (l(x)-l(y))) => :logitmeandiff)
end

ytitle = "Model - Null Model Accuracy (logit scale)"
pl = classdiffs |>
    @vlplot(
        facet = {column = {field = :target_time_label, title = nothing}},
        title = ["Low/High Salience Classification ","Accuracy by Target Time"],
        config = {legend = {disable = true}}
    ) + (
        @vlplot(color = {:condition, title = nothing}, x = {:condition, title = nothing}) +
        @vlplot(:bar,
            y = {:logitmeandiff, aggregate = :mean, type = :quantitative, title = ytitle}
        ) +
        @vlplot(:errorbar,
            color = {value = "black"},
            y = {:logitmeandiff, aggregate = :ci, type = :quantitative, title = ytitle}
        )
    );
pl |> save(joinpath(dir, "salience_earlylate.svg"))

# Plot 4-salience-level, trial-by-trial timeline
# =================================================================

# Compute frequency bins
# -----------------------------------------------------------------

classdf_sal4_timeline_file = joinpath(cache_dir("features"), "salience-4level-freqmeans-timeline-trial.csv")

if isfile(classdf_sal4_timeline_file)
    classdf_sal4_timeline = CSV.read(classdf_sal4_timeline_file)
else
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    classdf_sal4_timeline_groups = @_ events |>
        transform!(__, AsTable(:) => ByRow(x -> ishit(x, region = "target")) => :hittype) |>
        filter(_.hittype ∈ ["hit", "miss"], __) |>
        filter(_.condition == "global", __) |>
        groupby(__, [:sid, :condition, :salience_4level, :trial, :hittype])
    winbounds(start,k) = sid -> (start = start, len = winlen_bysid(sid) |>
        GermanTrack.spread(0.5,n_winlens,indices=k))

    windows = [winbounds(st,k) for st in range(0, 3, length = 64) for k in 1:n_winlens]
    classdf_sal4_timeline = compute_freqbins(subjects, classdf_sal4_timeline_groups,
        windowtarget, windows)

    CSV.write(classdf_sal4_timeline_file, classdf_sal4_timeline)
end

# Compute classification accuracy
# -----------------------------------------------------------------

# QUESTION: should we reselect lambda?

λsid = groupby(final_λs, :sid)

resultdf_sal4_timeline_file = joinpath(cache_dir("models"), "salience-timeline-sal4.csv")
classdf_sal4_timeline[!,:fold] = in.(classdf_sal4_timeline.sid, Ref(Set(λ_folds[1][1]))) .+ 1

levels = ["lowest","low","high","highest"]
classdf_sal4_timeline[!,:salience_4label] =
    CategoricalArray(get.(Ref(levels),coalesce.(classdf_sal4_timeline.salience_4level,0),missing), levels = levels)

modeltype = [ "1v4" => [1,4], "1v3" => [1,3], "1v2" => [1,2] ]

if isfile(resultdf_sal4_timeline_file) && mtime(resultdf_sal4_timeline_file) > mtime(classdf_sal4_timeline_file)
    resultdf_sal4_timeline = CSV.read(resultdf_sal4_timeline_file)
else
    factors = [:fold, :winlen, :winstart, :condition, :hittype]
    groups = groupby(classdf_sal4_timeline, factors)

    progress = Progress(length(groups) * length(modeltype))

    function findclass(((key, sdf), (type, levels)))
        λ = first(λsid[(sid = first(sdf.sid),)].λ)
        filtered = @_ sdf |> filter(_.salience_4level ∈ levels, __)

        result = testclassifier(LassoPathClassifiers([1.0, λ]),
            data = filtered, y = :salience_4label, X = r"channel", crossval = :sid,
            n_folds = n_folds, seed = stablehash(:salience_classification, 2019_11_18),
            maxncoef = size(filtered[:,r"channel"], 2),
            ycoding = StatsModels.SeqDiffCoding,
            irls_maxiter = 600, weight = :weight, on_model_exception = :throw)

        result[!, keys(key)]  .= permutedims(collect(values(key)))
        result[!, :modeltype] .= type
        next!(progress)

        result
    end

    resultdf_sal4_timeline = @_ groups |> pairs |> collect |>
        Iterators.product(__, modeltype) |>
        # foldl(append!!, Map(findclass), __)
        foldxt(append!!, Map(findclass), __)

    ProgressMeter.finish!(progress)
    CSV.write(resultdf_sal4_timeline_file, resultdf_sal4_timeline)

    alert("Completed salience timeline classification!")
end

# Display classification timeline
# -----------------------------------------------------------------

classmeans = @_ resultdf_sal4_timeline |>
    groupby(__, [:winstart, :winlen, :sid, :λ, :fold, :condition, :modeltype, :hittype]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :mean)

classmeans_sum = @_ classmeans |>
    groupby(__, [:winstart, :sid, :λ, :fold, :condition, :modeltype, :hittype]) |>
    combine(__, :mean => mean => :mean)

nullmeans = @_ classmeans_sum |>
    filter(_.λ == 1.0, __) |>
    rename!(__, :mean => :nullmean) |>
    deletecols!(__, :λ)

sallevel_pairs = [
    "1v4" => "High:\t1st vs. 4th Quartile",
    "1v3" => "Medium:\t1st vs. 3rd Quartile",
    "1v2" => "Low:\t1st vs. 2nd Quartile",
]
sallevel_shortpairs = [
    "1v4" => ["High" , "1st vs. 4th Q"],
    "1v3" => ["Medium", "1st vs. 3rd Q"],
    "1v2" => ["Low", "1st vs. 2nd Q"],
]

nullmean, classdiffs = let l = logit ∘ shrinktowards(0.5, by = 0.01), C = mean(l.(nullmeans.nullmean))
    100logistic(C),
    @_ classmeans_sum |>
        filter(_.λ != 1.0, __) |>
        innerjoin(__, nullmeans, on = [:winstart, :condition, :sid, :fold, :modeltype, :hittype]) |>
        transform!(__, [:mean, :nullmean] => ByRow((x,y) -> l(x) - l(y)) => :logitmeandiff) |>
        transform!(__, [:mean, :nullmean] => ByRow((x,y) -> 100logistic(l(x) - l(y) + C)) => :meancor) |>
        transform!(__, :condition => ByRow(uppercasefirst) => :condition) |>
        transform!(__, :modeltype => (x -> replace(x, sallevel_pairs...)) => :modeltype_title) |>
        transform!(__, :modeltype => (x -> replace(x , sallevel_shortpairs...)) => :modeltype_shorttitle)
end

annotate = @_ map(abs(_ - 3.0), classdiffs.winstart) |> classdiffs.winstart[argmin(__)]
ytitle = ["% Correct", "(Null-Model Corrected)"]
target_length_y = 70.0
pl = classdiffs |>
    @vlplot(
        title = {text = "Salience Classification Accuracy", subtitle = "Global Condition"},
        config = {
            legend = {orient = :none, legendX = 5, legendY = 0.5, title = "Classification"},
        },
        facet = {column = {field = :hittype, type = :nominal,
                 title = nothing}}
    ) +
    (
        @vlplot(color = {field = :modeltype_title, type = :nominal,
            scale = {scheme = :inferno}, sort = getindex.(sallevel_pairs, 2)}) +
        # data lines
        @vlplot(:line,
            x = {:winstart, type = :quantitative, title = "Time (s) Relative to Target Onset"},
            y = {:meancor, aggregate = :mean, type = :quantitative, title = ytitle,
                 scale = {domain = [50, 100]}
                }) +
        # data errorbands
        @vlplot(:errorband,
            x = {:winstart, type = :quantitative, title = "Time (s) Relative to Target Onset"},
            y = {:meancor, aggregate = :ci, type = :quantitative, title = ytitle}) +
        # condition labels
        @vlplot({:text, align = :left, dx = 5},
            transform = [{filter = "datum.winstart > 2.4 && datum.winstart < 2.5"}],
            x = {datum = 3.0},
            y = {:meancor, aggregate = :mean, type = :quantitative},
            text = :modeltype_shorttitle
        ) +
        # Basline (0 %) dotted line
        (
            @vlplot(data = {values = [{}]}) +
            @vlplot(mark = {:rule, strokeDash = [4 4], size = 2},
                y = {datum = nullmean},
                color = {value = "black"})
        ) +
        # "Target Length" arrow annotation
        (
            @vlplot(data = {values = [
                {x = 0.05, y = target_length_y, dir = 270},
                {x = 0.95, y = target_length_y, dir = 90}]}) +
            @vlplot(mark = {:line, size = 1.5},
                x = {:x, type = :quantitative},
                y = {:y, type = :quantitative},
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
                x = {datum = 0.5}, y = {datum = target_length_y},
                text = {value = "Target Length"},
                color = {value = "black"}
            )
        ) +
        # "Null Model" text annotation
        (
            @vlplot(data = {values = [{}]}) +
            @vlplot(mark = {:text, size = 11, baseline = "line-top", dy = 4},
                x = {datum = 2.0}, y = {datum = nullmean},
                text = {value = ["Mean Accuracy", "of Null Model"]},
                color = {value = "black"}
            )
        )
    );
pl |> save(joinpath(dir, "salience_timeline_4level.svg"))

hitvmiss = @_ classdiffs |>
    unstack(__, [:winstart, :sid, :condition, :modeltype], :hittype, :logitmeandiff) |>
    transform!(__, [:hit, :miss] => (-) => :hitvmiss) |>
    transform!(__, :modeltype => (x -> replace(x, sallevel_pairs...)) => :modeltype_title) |>
    transform!(__, :modeltype => (x -> replace(x , sallevel_shortpairs...)) => :modeltype_shorttitle)

annotate = @_ map(abs(_ - 3.0), classdiffs.winstart) |> classdiffs.winstart[argmin(__)]
ytitle = "Hit Accuracy - Miss Accuracy"
pl = hitvmiss |>
    @vlplot(
        # config = {legend = {disable = true}},
        config = {
            legend = {orient = :none, legendX = 5, legendY = 0.5, title = "Classification"},
        },
        title = ["Low/High Salience Classification Accuracy", "Object Condition"]
        # facet = {column = {field = :hittype, type = :nominal}}
    ) +
    (@vlplot(
        color = {field = :modeltype_title, type = :nominal, scale = {scheme = :inferno}},
    ) +
    # data lines
    @vlplot(:line,
        x = {:winstart, type = :quantitative, title = "Time (s) Relative to Target Onset"},
        y = {:hitvmiss, aggregate = :mean, type = :quantitative, title = ytitle}) +
    # data errorbands
    @vlplot(:errorband,
        x = {:winstart, type = :quantitative, title = "Time (s) Relative to Target Onset"},
        y = {:hitvmiss, aggregate = :ci, type = :quantitative, title = ytitle}) +
    # condition labels
    @vlplot({:text, align = :left, dx = 5},
        transform = [{filter = "datum.winstart > 2.05 && datum.winstart < 2.11"}],
        x = {datum = 3.0},
        y = {:hitvmiss, aggregate = :mean, type = :quantitative},
        text = :modeltype_shorttitle
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
pl |> save(joinpath(dir, "salience_timeline_4level_hitvmiss.svg"))

