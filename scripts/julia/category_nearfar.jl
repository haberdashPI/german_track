# Setup
# =================================================================

n_winlens = 12
n_folds = 10

using DrWatson; @quickactivate("german_track")
using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures, Dates, Underscores,
    Printf, ProgressMeter, VegaLite, FileIO, StatsBase, BangBang, Transducers,
    Infiltrator, Peaks, Distributions, DSP, Random, CategoricalArrays, StatsModels,
    StatsFuns, CSV, Colors, DataFramesMeta

dir = mkpath(plotsdir("figure4_parts"))

using GermanTrack: neutral, colors, lightdark, darkgray, inpatterns

# Find hyperparameters (λ and winlen)
# =================================================================

nbins = 10
switchbreaks = @_ GermanTrack.load_stimulus_metadata().switch_distance |>
    skipmissing |>
    quantile(__, range(0,1,length = nbins+1)[2:(end-1)])

file = joinpath(processed_datadir("analyses"), "nearfar-hyperparams.json")
GermanTrack.@cache_results file fold_map hyper_params begin
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")
    lambdas = 10.0 .^ range(-2, 0, length=100)

    function compute_powerbin_withbreak(sdf, switch_break, target_switch, window)
        filtered = @_ sdf |>
            filter(!ismissing(_.switch_class), __) |>
            filter(target_switch == "near" ? (_.switch_class < switch_break) :
                (_.switch_class >= switch_break), __)
        isempty(filtered) && return Empty(DataFrame)
        compute_powerbin_features(filtered, subjects, window)
    end

    classdf = @_ events |> filter(ishit(_) == "hit", __) |>
        @transform(__, switch_class = map(x -> sum(x .> switchbreaks).+1, :switch_distance)) |>
        groupby(__, [:sid, :condition, :target_time_label]) |>
        filteringmap(__, desc = "Computing features...",
            :switch_break => 2:nbins,
            :target_switch_label => ["near", "far"],
            :windows => [windowtarget(len = len, start = start)
                for len in 2.0 .^ range(-1, 1, length = 10),
                    start in [0; 2.0 .^ range(-2, 2, length = 10)]],
            compute_powerbin_withbreak)

    resultdf = @_ classdf |>
        addfold!(__, 2, :sid, rng = stableRNG(2019_11_18, :nearfar_hyper_folds)) |>
        mapgroups(__, folder = foldxt, desc = "Evaluating hyperparameters...",
            [:winstart, :winlen, :fold, :condition, :switch_break],
            function(sdf)
                testclassifier(LassoPathClassifiers(lambdas),
                    data         = sdf,
                    y            = :target_switch_label,
                    X            = r"channel",
                    crossval     = :sid,
                    n_folds      = 10,
                    seed         = 2017_09_16,
                    weight       = :weight,
                    maxncoef     = size(sdf[:, r"channel"], 2),
                    irls_maxiter = 1200,
                    on_missing_case = :missing,
                )
            end) |>
        deletecols!(__, :windows)

    fold_map = @_ resultdf |>
        groupby(__, :sid) |> combine(__, :fold => first => :fold) |>
        Dict(row.sid => row.fold for row in eachrow(__))

    hyper_params =
        filteringmap(resultdf, desc = nothing, folder = foldl,
            :fold => cross_folds(1:2),
            function (sdf, fold)
                λ = pick_λ(sdf, [:condition, :winstart, :winlen, :switch_break],
                    :condition, diffplot = "diffs_fold$fold",
                    smoothing = 0.85, slope_thresh_quantile = 0.95,
                    flat_thresh_ratio = 0.1, dir = joinpath(dir, "supplement"))

                factors = [:winlen, :winstart, :λ, :target_time_label, :switch_break]
                means = @_ sdf |>
                    @where(__, :λ .∈ Ref([1.0, λ])) |>
                    groupby(__, vcat(factors, :sid)) |>
                    @combine(__, mean = GermanTrack.wmean(:correct, :weight)) |>
                    groupby(__, factors) |>
                    @combine(__, mean = mean(:mean)) |>
                    groupby(__, setdiff(factors, [:λ])) |>
                    @transform(__,
                        logitnullmean = logit(shrink(only(:mean[:λ .== 1.0]))),
                        logitmean = logit.(shrink.(:mean))
                    ) |>
                    @where(__, :λ .!= 1.0) |>
                    groupby(__, setdiff(factors, [:target_time_label])) |>
                    @combine(__, score = mean(:logitmean - :logitnullmean))

                means[[argmax(means.score)],:]
            end)

    hyper_params = Dict(row.fold => NamedTuple(row[Not(:fold)])
        for row in eachrow(hyper_params))

    @info "Saving plots to $(joinpath(dir, "supplement"))"
end

# Plot near/far across early/late (Fig 4c)
# =================================================================

# Classification accuracy
# -----------------------------------------------------------------

file = joinpath(cache_dir("features"), "nearfar-target.json")
GermanTrack.@cache_results file resultdf begin
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    classdf = @_ events |> filter(ishit(_) == "hit", __) |>
        transform!(__, :sid => ByRow(x -> fold_map[x]) => :fold) |>
        transform!(__, [:fold, :switch_distance] =>
            ByRow((f,d) -> ismissing(d) ? missing :
                (d >= switchbreaks[hyper_params[f].switch_break] ? "near" : "far")) => :target_switch_label) |>
        groupby(__, [:sid, :fold, :condition, :target_time_label, :target_switch_label]) |>
        filteringmap(__, desc = "Computing features...",
            :window => [windowtarget(windowfn = event -> (
                start = start,
                len = hyper_params[event.fold[1]].winlen |>
                    GermanTrack.spread(0.5,n_winlens,indices=k)
            )) for start in [0; 2.0 .^ range(-2, 2, length = 10)] for k in 1:n_winlens],
            compute_powerbin_features(_1, subjects, _2)) |>
        deletecols!(__, :window)

    resultdf = @_ classdf |>
        transform!(__, :sid => ByRow(x -> fold_map[x]) => :fold) |>
        mapgroups(__, folder = foldxt, desc = "Classifying target proximity...",
            [:winstart, :winlen, :fold, :condition],
            function(sdf)
                testclassifier(LassoPathClassifiers([1.0, hyper_params[sdf.fold[1]].λ]),
                    data         = sdf,
                    y            = :target_switch_label,
                    X            = r"channel",
                    crossval     = :sid,
                    n_folds      = 10,
                    seed         = 2017_09_16,
                    weight       = :weight,
                    maxncoef     = size(sdf[:, r"channel"], 2),
                    irls_maxiter = 1200
                )
            end)

    # GermanTrack.@store_cache file resultdf
end

# Plot data
# -----------------------------------------------------------------

# classmeans = @_ hyper_resultdf |>
#     filter(_.switch_break == 3, __) |>
#     filter(_.λ ∈ [1.0, λ_map[_.fold]], __) |>
classmeans = @_ resultdf |>
    groupby(__, [:winstart, :winlen, :sid, :λ, :fold, :condition, :target_time_label]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :mean)


classmeans_sum = @_ classmeans |>
    groupby(__, [:winstart, :sid, :λ, :fold, :condition, :target_time_label]) |>
    @combine(__, mean = maximum(:mean)) |>
    groupby(__, [:sid, :λ, :fold, :condition, :target_time_label]) |>
    @combine(__, mean = mean(:mean)) |>
    transform!(__, :λ => ByRow(log) => :logλ)

statdata = @_ classmeans_sum |>
    groupby(__, [:condition, :sid, :target_time_label]) |>
    @transform(__,
        logitnullmean = logit(shrink(only(:mean[:λ .== 1.0]))),
        logitmean     = logit.(shrink.(:mean)),
    ) |>
    @where(__, :λ .!= 1.0)

pl = statdata |> @vlplot(:point,
    column = :condition,
    color = :target_time_label,
    x     = :logitnullmean,
    y     = :logitmean,
);
pl |> save(joinpath(dir, "supplement", "earlylate_nearfar_ind.svg"))

CSV.write(processed_datadir("analyses", "eeg_nearfar.csv"), statdata)

run(`Rscript $(joinpath(scriptsdir("R"), "nearfar_eeg.R"))`)

plotdata = @_ CSV.read(processed_datadir("analyses", "eeg_nearfar_coefs.csv"), DataFrame) |>
    rename(__, :r_med => :mean, :r_05 => :lower, :r_95 => :upper) |>
    @transform(__,
        condition_time = :condition,
        condition = getindex.(split.(:condition, "_"),1),
        target_time_label = getindex.(split.(:condition, "_"),2)
    )
nullmean = logistic(mean(statdata.logitnullmean))
ytitle = ["Switch Proximity (Near/Far)", "Classification"]
barwidth = 14
yrange = [0.5, 1]
pl = plotdata |>
    @vlplot(
        height = 175, width = 242, autosize = "fit",
        config = {
            legend = {disable = true},
            bar = {discreteBandSize = barwidth},
            axis = {titlePadding = 13}
        },
    ) +
    @vlplot({:bar, xOffset = -(barwidth/2), clip = true},
        transform = [{filter = "datum.target_time_label == 'early'"}],
        color = {:condition_time, title = nothing,
            scale = {range = urlcol.(keys(inpatterns))}},
        x = {:condition, axis = {title = "", labelAngle = 0,
            labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}},
        y = {:mean, title = ytitle, scale = {domain = yrange}}
    ) +
    @vlplot({:rule, xOffset = -(barwidth/2)},
        transform = [{filter = "datum.target_time_label == 'early'"}],
        color = {value = "black"},
        x = {:condition, title = nothing},
        y = {:lower, title = ""}, y2 = :upper
    ) +
    @vlplot({:bar, xOffset = (barwidth/2), clip = true},
        transform = [{filter = "datum.target_time_label == 'late'"}],
        color = {:condition_time, title = nothing},
        x = {:condition, title = nothing},
        y = {:mean, title = ytitle}
    ) +
    @vlplot({:rule, xOffset = (barwidth/2)},
        transform = [{filter = "datum.target_time_label == 'late'"}],
        x = {:condition, title = nothing},
        color = {value = "black"},
        y = {:lower, title = ""}, y2 = :upper
    ) +
    @vlplot({:text, angle = -90, fontSize = 9, align = "left", baseline = "bottom", dx = 0, dy = -barwidth-2},
        transform = [{filter = "datum.target_time_label == 'early' && datum.condition == 'global'"}],
        # x = {datum = "spatial"}, y = {datum = 0.},
        x = {:condition, axis = {title = ""}},
        y = {datum = yrange[1]},
        text = {value = "Early"},
    ) +
    @vlplot({:text, angle = -90, fontSize = 9, align = "left", baseline = "top", dx = 0, dy = barwidth+2},
        transform = [{filter = "datum.target_time_label == 'late' && datum.condition == 'global'"}],
        # x = {datum = "spatial"}, y = {datum = },
        x = {:condition, axis = {title = ""}},
        y = {datum = yrange[1]},
        text = {value = "Late"},
    ) +
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:rule, strokeDash = [4 4], size = 2},
            y = {datum = nullmean},
            color = {value = "black"}) +
        @vlplot({:text, align = "left", dx = 12, dy = 0, baseline = "line-bottom", fontSize = 9},
            y = {datum = nullmean},
            x = {datum = "spatial"},
            text = {value = ["Null Model", "Accuracy"]}
        )
    );
plotfile = joinpath(dir, "fig4c.svg")
pl |> save(plotfile)
addpatterns(plotfile, inpatterns, size = 10)
