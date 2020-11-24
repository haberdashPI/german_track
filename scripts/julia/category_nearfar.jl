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

using GermanTrack: neutral, colors, lightdark, darkgray

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
        mapgroups(__, folder = foldl, desc = "Evaluating hyperparameters...",
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
                    irls_maxiter = 600
                )
            end)

    fold_map = @_ resultdf |>
        groupby(__, :sid) |> combine(__, :fold => first => :fold) |>
        Dict(row.sid => row.fold for row in eachrow(__))

    λ_map, winlen_map = pick_λ_winlen(resultdf,
        [:condition, :sid, :winstart, :switch_break], :condition,
        smoothing = 0.85, slope_thresh = 0.15, flat_thresh = 0.01,
        dir = mkpath(joinpath(dir, "supplement")))

    classmeans_sum = @_ resultdf |>
        groupby(__, [:winstart, :winlen, :sid, :λ, :fold, :condition, :target_time_label,
            :switch_break]) |>
        combine(__, [:correct, :weight] => GermanTrack.wmean => :mean) |>
        groupby(__, [:sid, :λ, :fold, :condition, :target_time_label, :switch_break]) |>
        combine(__, :mean => mean => :mean)

    nullmeans = @_ classmeans_sum |>
        filter(_.λ == 1.0, __) |>
        rename!(__, :mean => :nullmean) |>
        deletecols!(__, :λ)

    break_map = @_ classmeans_sum |>
        filter(_.λ != 1.0, __) |>
        innerjoin(__, nullmeans, on = [:condition, :sid, :fold, :target_time_label, :switch_break]) |>
        @transform(__, logitnullmean = logit.(shrinktowards.(:nullmean, 0.5, by = 0.01)),
                       logitmean = logit.(shrinktowards.(:mean, 0.5, by = 0.01))) |>
        groupby(__, [:switch_break, :fold]) |>
        @based_on(__,
            mean = mean(:logitmean .- :logitnullmean),
            count = length(unique(string.(:condition, :target_time_label)))
        ) |>
        filter(_.count == 6, __) |> # ignore conditions with missing data
        filteringmap(__,
            :train_fold => map(fold -> fold => (sdf -> sdf.fold != fold), unique(__.fold)),
            (sdf, fold) -> DataFrame(best = sdf.switch_break[argmax(sdf.mean)])) |>
        @transform(__, early_break = switchbreaks[:best .+ 1]) |>
        Dict(r.train_fold => r.best for r in eachrow(__))

    @info "Saving plots to $(joinpath(dir, "supplement"))"
end

# Plot near/far across early/late (Fig 4c)
# =================================================================

# Classification accuracy
# -----------------------------------------------------------------

file = joinpath(cache_dir("features"), "nearfar-target.json")
GermanTrack.@cache_results file resultdf begin
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    classdf = compute_freqbins(
        subjects = subjects,
        groupdf  = @_( events |>
            transform!(__, AsTable(:) => ByRow(ishit) => :hittype) |>
            transform!(__, :sid => ByRow(x -> fold_map[x]) => :fold) |>
            filter(_.hittype ∈ ["hit", "miss"], __) |>
            groupby(__, [:sid, :fold, :condition, :salience_label, :hittype])
        ),
        windows = [
            windowtarget(windowfn = event -> (
                start = start,
                len = winlen_map[event.fold[1]] |>
                    GermanTrack.spread(0.5,n_winlens,indices=k)))
            for start in range(0, 3, length = 32) for k in 1:n_winlens
        ]
    )

    resultdf = @_ classdf |>
        mapgroups(__, [:winlen, :fold, :hittype, :condition], desc = "Classifying salience...",
            function (sdf)
                result = testclassifier(LassoPathClassifiers([1.0, λ_map[sdf.fold[1]]]),
                    data         = sdf,
                    y            = :salience_label,
                    X            = r"channel",
                    crossval     = :sid,
                    n_folds      = n_folds,
                    seed         = stablehash(:salience_classification, 2019_11_18),
                    maxncoef     = size(sdf[:, r"channel"], 2),
                    irls_maxiter = 600,
                    weight       = :weight)
            end)
end



