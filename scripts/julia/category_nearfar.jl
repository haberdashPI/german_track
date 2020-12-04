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
GermanTrack.@cache_results file fold_map λ_map winlen_map break_map begin
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")
    lambdas = 10.0 .^ range(-2, 0, length=100)

    function compute_powerbin_withbreak(sdf, switch_break, target_switch, window)
        filtered = @_ sdf |>
            filter(!ismissing(_.switch_class), __) |>
            filter(target_switch == "far" ? (_.switch_class > switch_break) :
                (_.switch_class <= switch_break), __)
        isempty(filtered) && return Empty(DataFrame)
        compute_powerbin_features(filtered, subjects, window)
    end

    classdf = @_ events |> filter(ishit(_) == "hit", __) |>
        @transform(__, switch_class = map(x -> sum(x .> switchbreaks), :switch_distance)) |>
        groupby(__, [:sid, :condition, :target_time_label]) |>
        filteringmap(__, desc = "Computing features...",
            :switch_break => 1:(nbins-2),
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
                    irls_maxiter = 1200
                )
            end) |>
        deletecols!(__, :windows)

    fold_map = @_ resultdf |>
        groupby(__, :sid) |> combine(__, :fold => first => :fold) |>
        Dict(row.sid => row.fold for row in eachrow(__))

    resultdf |>
        filteringmap(__, descr = nothing, folder = foldl,
            :train_fold = cross_folds(1:2),
            function (sdf, fold)
                λ = pick_λ(sdf, [:condition, :winstart, :winlen, :switch_break],
                    :condition, diffplot = "diffs_fold$fold", lambda
                    smoothing = 0.85, slope_thresh_quantile = 0.95,
                    flat_thresh_ratio = 0.1)
                winlen = 0 # TODO: stopped here
            end
        )

    λ_map, winlen_map = pick_λ_winlen(resultdf,
        [:condition, :sid, :winstart, :switch_break], :condition,
        smoothing = 0.85, slope_thresh = 0.15, flat_thresh = 0.05,
        dir = mkpath(joinpath(dir, "supplement")))

    classmeans_sum = @_ resultdf |>
        groupby(__, [:winstart, :winlen, :sid, :λ, :fold, :condition, :target_time_label,
            :switch_break]) |>
        @combine(__, mean = GermanTrack.wmean(:correct, :weight)) |>
        groupby(__, [:sid, :λ, :fold, :condition, :target_time_label, :switch_break]) |>
        @combine(__, mean = mean(:mean))

    break_diffs = @_ classmeans_sum |>
        groupby(__, [:sid, :condition, :target_time_label, :fold, :switch_break]) |>
        @transform(__, logitnullmean = logit(shrinktowards(only(:mean[:λ .== 1.0]), 0.5, by = 0.01)),
                       logitmean = logit.(shrinktowards.(:mean, 0.5, by = 0.01)),
                       early_break = switchbreaks[:switch_break .+ 1]) |>
        filter(_.λ != 1.0, __)

    break_map_df = @_ break_diffs |>
        groupby(__, [:switch_break, :fold]) |>
        @combine(__,
            mean = mean(:logitmean .- :logitnullmean),
            count = length(unique(string.(:condition, :target_time_label)))
        ) |>
        filter(_.count == 6, __) |> # ignore conditions with missing data
        filteringmap(__,
            :train_fold => map(fold -> fold => (sdf -> sdf.fold != fold), unique(__.fold)),
            (sdf, fold) -> DataFrame(best = sdf.switch_break[argmax(sdf.mean)])) |>
        @transform(__, early_break = switchbreaks[:best .+ 1])

    break_map = @_ break_map_df |>
        Dict(r.train_fold => r.early_break for r in eachrow(__))

    @_ break_diffs |>
        groupby(__, [:condition, :target_time_label, :switch_break, :early_break]) |>
        @combine(__,
            mean  = mean(:logitmean .- :logitnullmean),
            lower = lowerboot(:logitmean .- :logitnullmean),
            upper = upperboot(:logitmean .- :logitnullmean),
        ) |>
        @vlplot(
            facet = {row = {field = :condition, title = ""}},
        ) + (
            @vlplot() +
            @vlplot(:line,
                color = :target_time_label,
                x = {:early_break, title = "Near/Far Divide (s)", scale = {domain = [0, 2.5]}},
                y = {:mean, title = ["Classsification Accuracy", "(Null Mean Corrected)"]}
            ) +
            @vlplot(:point,
                color = :target_time_label,
                x = :early_break,
                y = {:mean, title = ""}
            ) +
            @vlplot(:errorband,
                color = :target_time_label,
                x = :early_break,
                y = {:lower, title = ""}, y2 = :upper
            ) + (
                break_map_df |> @vlplot() +
                @vlplot({:rule, strokeDash = [2,2]},
                    x = :early_break,
                ) +
                @vlplot({:text, align = :left, fontSiz = 9, xOffset = 2},
                    transform = [{calculate = "'Fold '+datum.train_fold", as = :fold_label}],
                    x = :early_break,
                    y = {datum = 0.2},
                    text = :fold_label
                )
            )
        ) |> save(joinpath(dir, "supplement", "switch_target_earlylate_multibreak.svg"))

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
                (d >= break_map[f] ? "near" : "far")) => :target_switch_label) |>
        groupby(__, [:sid, :fold, :condition, :target_time_label, :target_switch_label]) |>
        filteringmap(__, desc = "Computing features...",
            :window => [windowtarget(windowfn = event -> (
                start = start,
                len = winlen_map[event.fold[1]] |>
                    GermanTrack.spread(0.5,n_winlens,indices=k)
            )) for start in [0; 2.0 .^ range(-2, 2, length = 10)] for k in 1:n_winlens],
            compute_powerbin_features(_1, subjects, _2)) |>
        deletecols!(__, :window)

    resultdf = @_ classdf |>
        transform!(__, :sid => ByRow(x -> fold_map[x]) => :fold) |>
        mapgroups(__, folder = foldxt, desc = "Classifying target proximity...",
            [:winstart, :winlen, :fold, :condition],
            function(sdf)
                testclassifier(LassoPathClassifiers([1.0, λ_map[sdf.fold[1]]]),
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

classmeans = @_ hyper_resultdf |>
    filter(_.switch_break == 3, __) |>
    filter(_.λ ∈ [1.0, λ_map[_.fold]], __) |>
# classmeans = @_ resultdf |>
    groupby(__, [:winstart, :winlen, :sid, :λ, :fold, :condition, :target_time_label]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :mean)


classmeans_sum_time = @_ classmeans |>
    groupby(__, [:winstart, :sid, :λ, :fold, :condition, :target_time_label]) |>
    @combine(__, mean = maximum(:mean)) |>
    transform!(__, :λ => ByRow(log) => :logλ)

classmeans_sum = @_ classmeans |>
    groupby(__, [:winstart, :sid, :λ, :fold, :condition, :target_time_label]) |>
    @combine(__, mean = maximum(:mean)) |>
    groupby(__, [:sid, :λ, :fold, :condition, :target_time_label]) |>
    @combine(__, mean = mean(:mean)) |>
    transform!(__, :λ => ByRow(log) => :logλ)

nullmeans = @_ classmeans_sum |>
    filter(_.λ == 1.0, __) |>
    rename!(__, :mean => :nullmean) |>
    deletecols!(__, [:λ, :logλ])

nullmeans_time = @_ classmeans_sum_time |>
    filter(_.λ == 1.0, __) |>
    rename!(__, :mean => :nullmean) |>
    deletecols!(__, [:λ, :logλ])

statdata = @_ classmeans_sum |>
    filter(_.λ != 1.0, __) |>
    innerjoin(__, nullmeans, on = [:condition, :sid, :fold, :target_time_label]) |>
    @transform(__,
        logitnullmean = logit.(shrinktowards.(:nullmean, 0.5, by = 0.01)),
        logitmean     = logit.(shrinktowards.(:mean, 0.5, by = 0.01)),
    )

statdata_time = @_ classmeans_sum_time |>
    filter(_.λ != 1.0, __) |>
    innerjoin(__, nullmeans_time, on = [:winstart, :condition, :sid, :fold, :target_time_label]) |>
    @transform(__,
        logitnullmean = logit.(shrinktowards.(:nullmean, 0.5, by = 0.01)),
        logitmean     = logit.(shrinktowards.(:mean, 0.5, by = 0.01)),
    ) |>
    @transform(__,
        logitdiff     = :logitmean .- :logitnullmean
    )

pl = statdata_time |> @vlplot(
    facet = {column = {field = :condition, type = :nominal}}
) +
(
    @vlplot(
        x = :winstart,
        color = :target_time_label
    ) +
    @vlplot(:line, y = {:logitdiff, aggregate = :mean, type = :quantitative}) +
    @vlplot(:errorband, y = {:logitdiff, aggregate = :ci, type = :quantitative})
);
pl |> save(joinpath(dir, "supplement", "earlylate_nearfar_winlen.svg"))

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
