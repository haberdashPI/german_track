# Setup
# =================================================================

n_winlens = 12
n_folds = 10

using DrWatson; @quickactivate("german_track")
using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures, Dates, Underscores,
    Printf, ProgressMeter, VegaLite, FileIO, StatsBase, BangBang, Transducers,
    Infiltrator, Peaks, Distributions, DSP, Random

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

# display lambdas
# -----------------------------------------------------------------

means = @_ resultdf |>
    groupby(__, [:condition, :λ, :nzcoef, :sid, :fold, :winstart, :winlen]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :mean)

bestmeans = @_ means |>
    groupby(__, [:condition, :λ, :nzcoef, :sid, :fold]) |>
    combine(__, :mean => maximum => :mean)

pl = @vlplot() +
vcat(
    bestmeans |> @vlplot(
        width = 750, height = 100,
        :line,
        color = {field = :condition, type = :nominal},
        x = {:λ, scale = {type = :log}},
        y = {:nzcoef, aggregate = :max, type = :quantitative}
    ),
    (
        bestmeans |> @vlplot(
            width = 750, height = 400,
            x = {:λ, scale = {type = :log}},
            color = {field = :condition, type = :nominal},
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

# Subtract null model to find peak in λ performance
# -----------------------------------------------------------------

meandiff = @_ filter(_.λ == 1.0, bestmeans) |>
    deletecols!(__, [:λ, :nzcoef]) |>
    rename!(__, :mean => :nullmean) |>
    innerjoin(__, bestmeans, on = [:condition, :sid, :fold]) |>
    transform!(__, [:mean,:nullmean] => (-) => :meandiff)

grandmeandiff = @_ meandiff |>
    groupby(__, [:λ, :fold]) |>
    combine(__, :meandiff => mean => :meandiff) |>
    sort!(__, [:λ]) |>
    transform!(__, :meandiff => (x -> filtfilt(digitalfilter(Lowpass(0.1), Butterworth(5)), x)) => :meandiff)

pl = grandmeandiff |> @vlplot() +
    @vlplot(:line,
        x     = {:λ, scale = {type = :log, domain = [0.01, 0.35]},
        title = "Regularization Parameter (λ)"},
        y     = {:meandiff, aggregate = :mean, type = :quantitative,
                title = "# of non-zero coefficients (max)"})

pl |> save(joinpath(dir, "grandmean.svg"))

# Show final λ selection
# -----------------------------------------------------------------

# pick the largest valued λ, with a non-negative peak for meandiff
function pickλ(df)
    peaks = @_ maxima(df.meandiff) |>
        filter(df.meandiff[_] > 0.01, __)
    maxλ = argmax(df[peaks,:λ])
    df[peaks[maxλ],[:λ]]
end
λs = @_ grandmeandiff |> groupby(__,:fold) |> combine(pickλ,__)
λs[!,:fold_text] .= string.("Fold: ",λs.fold)
λs[!,:yoff] = [0.1,0.15]

pl = @vlplot() +
    vcat(
        meandiff |> @vlplot(
        :line, width = 750, height = 100,
            color = {field = :condition, type = :nominal},
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
                    color = {field = :condition, type = :nominal}) +
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

pl |> save(joinpath(dir, "salience_lambdas.svg"))
pl |> save(joinpath(dir, "salience_lambdas.png"))

final_λs = vcat((DataFrame(sid = sid, λ = λ, fold = fi)
    for (fi, (λ, λ_fold)) in enumerate(zip(λs.λ, λ_folds))
    for sid in λ_fold[2])...)

# Plot best lambda across window grid
# -----------------------------------------------------------------

λsid = groupby(final_λs, :sid)

windowmeans = @_ resultdf |>
    filter(_.λ ∈ (1.0, first(λsid[(sid = _.sid,)].λ)), __) |>
    groupby(__,[:condition, :sid, :fold, :λ, :winlen, :winstart]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :mean)

nullmeans = @_ windowmeans |>
    filter(_.λ == 1.0, __) |>
    deletecols!(__, :λ) |>
    rename!(__, :mean => :nullmean)

windowdiff = @_ windowmeans |>
    filter(_.λ != 1.0, __) |>
    innerjoin(nullmeans, __, on = [:condition, :sid, :fold, :winlen, :winstart]) |>
    transform!(__, [:mean, :nullmean] => (-) => :meandiff)

pl = windowdiff |>
    @vlplot(:rect,
        config =  {view = {stroke = :transparent}},
        column = :condition,
        # row = :fold,
        y = {:winlen, type = :ordinal, axis = {format = ".2f"}, sort = :descending,
            title = "Length (s)"},
        x = {:winstart, type = :ordinal, axis = {format = ".2f"}, title = "Start (s)"},
        color = {:meandiff, aggregate = :mean, type = :quantitative,
            scale = {scheme = "redblue", domainMid = 0}})

pl |> save(joinpath(dir, "salience_windows.svg"))
pl |> save(joinpath(dir, "salience_windows.png"))

# Plot timeline
# =================================================================

# Compute best window length
# -----------------------------------------------------------------

windavg = @_ windowdiff |> groupby(__, [:condition, :fold, :winlen, :winstart]) |>
    combine(__, :meandiff => mean => :meandiff) |>
    groupby(__, [:fold, :winlen, :condition]) |>
    combine(__, :meandiff => maximum => :meandiff)

pl = windavg |>
    @vlplot(:rect,
        config =  {view = {stroke = :transparent}},
        column = :condition,
        row = :fold,
        y = {:winlen, type = :ordinal, axis = {format = ".2f"}, sort = :descending,
            title = "Length (s)"},
        # x = {:winstart, type = :ordinal, axis = {format = ".2f"}, title = "Start (s)"},
        color = {:meandiff, aggregate = :mean, type = :quantitative,
            scale = {scheme = "redblue", domainMid = 0}})
pl |> save(joinpath(dir, "salience_winavg.svg"))

bestlens = @_ windavg |>
    groupby(__, [:winlen, :fold]) |>
    combine(__, :meandiff => mean => :meandiff) |>
    groupby(__, [:fold]) |>
    combine(__, [:meandiff, :winlen] =>
        ((m,l) -> l[argmax(m)]) => :winlen,
        :meandiff => maximum => :meandiff)

bestlen_bysid = @_ bestlens |>
    groupby(__, [:fold, :winlen, :meandiff]) |>
    combine(__, :fold => (f -> λ_folds[f |> first][2]) => :sid) |>
    groupby(__, :sid)
winlen_bysid(sid) = bestlen_bysid[(sid = sid,)].winlen |> first

# Compute frequency bins
# -----------------------------------------------------------------

classdf_timeline_file = joinpath(cache_dir("features"), "salience-freqmeans-timeline.csv")

if isfile(classdf_timeline_file)
    classdf_timeline = CSV.read(classdf_timeline_file)
else
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    classdf_timeline_groups = @_ events |>
        filter(ishit(_, region = "target") ∈ ["hit"], __) |>
        groupby(__, [:sid, :condition, :salience_label])
    winbounds(start,k) = sid -> (start = start, len = winlen_bysid(sid) |>
        GermanTrack.spread(0.5,n_winlens,k))

    windows = [winbounds(st,k) for st in range(0, 3, length = 64) for k in 1:n_winlens]
    classdf_timeline = compute_freqbins(subjects, classdf_timeline_groups, windowtarget, windows, foldl)

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
    factors = [:fold, :winlen, :winstart, :condition]
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
        foldxt(append!!, Map(findclass), __)

    ProgressMeter.finish!(progress)
    CSV.write(resultdf_timeline_file, resultdf_timeline)

    alert("Completed salience timeline classification!")
end

# Display classification timeline
# -----------------------------------------------------------------

classmeans = @_ resultdf_timeline |>
    groupby(__, [:winstart, :winlen, :sid, :λ, :fold, :condition]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :mean)

classmeans_sum = @_ classmeans |>
    groupby(__, [:winstart, :sid, :λ, :fold, :condition]) |>
    combine(__, :mean => mean => :mean)

nullmeans = @_ classmeans_sum |>
    filter(_.λ == 1.0, __) |>
    rename!(__, :mean => :nullmean) |>
    deletecols!(__, :λ)

classdiffs = @_ classmeans_sum |>
    innerjoin(__, nullmeans, on = [:winstart, :condition, :sid, :fold]) |>
    transform!(__, [:mean, :nullmean] => ((x,y) -> 100*(x-y)) => :meandiff)

annotate = @_ map(abs(_ - 3.0), classdiffs.winstart) |> classdiffs.winstart[argmin(__)]
ytitle = "% Correct - % Correct of Null"
pl = classdiffs |>
    @vlplot(
        title = "Low/High Salience Classification Accuracy",
        color = {field = :condition, type = :nominal},
        config = {legend = {disable = true}}
    ) +
    # data lines
    @vlplot(:line,
        x = {:winstart, type = :quantitative, title = "Time (s) Relative to Target Onset"},
        y = {:meandiff, aggregate = :mean, type = :quantitative, title = ytitle}) +
    # data errorbands
    @vlplot(:errorband,
        x = {:winstart, type = :quantitative, title = "Time (s) Relative to Target Onset"},
        y = {:meandiff, aggregate = :ci, type = :quantitative, title = ytitle}) +
    # condition labels
    @vlplot({:text, align = :left, dx = 5},
        transform = [{filter = "datum.winstart > 2.2"}],
        x = {datum = 3.0},
        y = {:meandiff, aggregate = :mean, type = :quantitative},
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
            {x = 0.05, y = 10, dir = 270},
            {x = 0.95, y = 10, dir = 90}]}) +
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
            x = {datum = 0.5}, y = {datum = 10},
            text = {value = "Target Length"},
            color = {value = "black"}
        )
    );
pl |> save(joinpath(dir, "salience_timeline.svg"))

# Early/late targets
# =================================================================

# Select best window time
# -----------------------------------------------------------------

timemean = @_ classdiffs |>
    groupby(__, [:fold, :winstart, :condition]) |>
    combine(__, :meandiff => mean => :meandiff) |>
    groupby(__, [:fold, :winstart]) |>
    combine(__, :meandiff => maximum => :meandiff) |>
    sort!(__, :winstart) |>
    groupby(__, [:fold]) |>
    transform!(__, :meandiff =>
        (x -> filtfilt(digitalfilter(Lowpass(0.5), Butterworth(5)), x)) => :meandiff)
pl = timemean |>
    @vlplot(:line, x = :winstart, y = :meandiff, row = {:fold, type = :nominal},
        color = :condition);
pl |> save(joinpath(dir, "salience_winstart_avg.svg"))

beststart = @_ timemean|>
    groupby(__, [:fold]) |>
    combine(__, [:meandiff, :winstart] =>
        ((m,l) -> l[argmax(m)]) => :winstart,
        :meandiff => maximum => :meandiff)

beststart_bysid = @_ beststart |>
    groupby(__, [:fold, :winstart, :meandiff]) |>
    combine(__, :fold => (f -> λ_folds[f |> first][2]) => :sid) |>
    groupby(__, :sid)
winstart_bysid(sid) = beststart_bysid[(sid = sid,)].winstart |> first

# Compute frequency bins
# -----------------------------------------------------------------

classdf_earlylate_file = joinpath(cache_dir("features"), "salience-freqmeans-earlylate-timeline.csv")

if isfile(classdf_earlylate_file)
    classdf_earlylate = CSV.read(classdf_earlylate_file)
else
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    classdf_earlylate_groups = @_ events |>
        filter(ishit(_, region = "target") ∈ ["hit"], __) |>
        groupby(__, [:sid, :condition, :salience_label, :target_time_label])
    winbounds(start,k) = sid -> (
        start = winstart_bysid(sid),
        len = winlen_bysid(sid) |> GermanTrack.spread(0.5,n_winlens,k)
    )

    windows = [winbounds(2.25,k) for k in 1:n_winlens]
    classdf_earlylate = compute_freqbins(subjects, classdf_earlylate_groups, windowtarget, windows, foldl)

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
    groupby(__, [:winstart, :sid, :λ, :fold, :condition, :target_time_label]) |>
    combine(__, :mean => mean => :mean)

nullmeans = @_ classmeans_sum |>
    filter(_.λ == 1.0, __) |>
    rename!(__, :mean => :nullmean) |>
    deletecols!(__, :λ)

classdiffs = @_ classmeans_sum |>
    innerjoin(__, nullmeans, on = [:winstart, :condition, :sid, :fold, :target_time_label]) |>
    transform!(__, [:mean, :nullmean] => ((x,y) -> 100*(x-y)) => :meandiff)

ytitle = "% Correct - % Correct of Null"
pl = classdiffs |>
    @vlplot(
        facet = {column = {field = :condition, title = nothing}},
        title = ["Low/High Classification ","Accuracy by Target Time"],
        config = {legend = {disable = true}}
    ) + (
        @vlplot(color = :condition, x = {:target_time_label, title = ["Target", "Time"]}) +
        @vlplot(:bar,
            y = {:meandiff, aggregate = :mean, type = :quantitative, title = ytitle}
        ) +
        @vlplot(:errorbar,
            color = {value = "black"},
            y = {:meandiff, aggregate = :ci, type = :quantitative, title = ytitle}
        )
    );
pl |> save(joinpath(dir, "salience_earlylate_timeline.svg"))

# Plot 4-salience-level timeline
# =================================================================

# TODO: setup new features using sal4 and then run classifier with
# different coding schemes (start with the sequential one)

# test out 3-class classifier
X = [[4, 0, 0]' .+ rand(Normal(0,0.1), 20, 3);
     [0, 4, 0]' .+ rand(Normal(0,0.1), 20, 3);
     [0, 0, 4]' .+ rand(Normal(0,0.1), 20, 3)];
y = [fill("A", 20); fill("B", 20); fill("C", 20)];
df = DataFrame(x1 = X[:,1], x2 = X[:,2], x3 = X[:,3], y = y, id = repeat(1:20, outer=3))

result = testclassifier(LassoClassifier(0.0), data = df, X = r"x[0-9]+", y = :y,
    n_folds = 5, crossval = :id, seed = stablehash(:test), irls_maxiter = 600, ycoding = DummyCoding)

# Compute frequency bins
