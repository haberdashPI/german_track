# Setup
# =================================================================

n_winlens = 12
n_folds = 10

using DrWatson
@quickactivate("german_track")
using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures, Dates, Underscores,
    Printf, ProgressMeter, VegaLite, FileIO, StatsBase, BangBang, Transducers,
    Infiltrator, Peaks, Distributions, DSP, Random, CategoricalArrays, StatsModels,
    StatsFuns, CSV, Colors, DataFramesMeta, Lasso, Bootstrap

dir = mkpath(plotsdir("figure4_parts"))

using GermanTrack: neutral, colors, lightdark, darkgray, inpatterns

# Behavior data
# -----------------------------------------------------------------

# nearsplit = mean(switchclass[collect(values(best_breaks))])

target_labels = OrderedDict(
    "early" => ["Early Target", "(before 3rd Switch)"],
    "late"  => ["Late Target", "(after 3rd Switch)"]
)

ascondition = Dict(
    "test" => "global",
    "feature" => "spatial",
    "object" => "object"
)

file = joinpath(raw_datadir("behavioral", "export_ind_data.csv"))
rawdata = @_ CSV.read(file, DataFrame) |>
    transform!(__, :block_type => ByRow(x -> ascondition[x]) => :condition) |>
    @where(__, 0.0 .< :direction_timing .< 1.0)

function find_switch_distance(time, switches, dev_dir)
    sel_switches = dev_dir == "right" ?
        switches[r"right_switches"] : switches[r"left_switches"]
    if time <= 0
        [missing, missing]
    else
        distances = time .- sort!(collect(skipmissing(Array(sel_switches))))
        closest = @_ findlast(!ismissing(_1) && _1 > 0, distances)
        if isnothing(closest)
            [missing, missing]
        else
            [distances[closest], closest]
        end
    end
end
rawdata.switch_index = 0
rawdata.switch_distance = 0.0
allowmissing!(rawdata, [:switch_index, :switch_distance])
rawdata[:,[:switch_distance, :switch_index]] =
    mapreduce(find_switch_distance, hcat,
        rawdata[!,:dev_time], eachrow(rawdata[!,r"switches"]), rawdata[!, :dev_direction])'

function extend(xs)
    nonmissing = missing
    ys = similar(xs)
    ys[1] = xs[1]
    for i in 2:length(xs)
        ys[i] = xs[i] !== missing ? xs[i] : ys[i-1]
    end
    ys
end

meansraw = @_ rawdata |>
    groupby(__, [:sid, :condition, :exp_id]) |>
    @combine(__,
        hr = sum(:perf .== "hit") / sum(:perf .∈ Ref(Set(["hit", "miss"]))),
        fr = sum(:perf .== "false") / sum(:perf .∈ Ref(Set(["false", "reject"]))),
    )

bad_sids = @_ meansraw |>
    @where(__, :condition .== "global") |>
    @where(__, (:hr .<= :fr) .| (:fr .>= 1)) |> __.sid |> Ref |> Set


nbins = 30
binwidth = 0.7
qs = quantile(skipmissing(rawdata.direction_timing), range(0, 1, length = nbins+1))
bin_means = (qs[1:(end-1)] + qs[2:end])/2
getids(df) = @_ df |> groupby(__, [:sid, :exp_id]) |>
    combine(first, __) |> zip(__.sid, __.exp_id)
allids = getids(rawdata)
hit_by_switch = @_ rawdata |>
    @where(__, :sid .∉ bad_sids) |>
    @transform(__, time_bin = cut(:direction_timing, nbins, allowempty = true), target_bin = cut(:dev_time, 2)) |>
    @transform(__, target_time_label =
        ifelse.(ismissing.(:switch_distance) .| (:switch_index .<= 1), "early", "late")) |>
    @transform(__,
        time_bin_mean = Array(recode(:time_bin, (levels(:time_bin) .=> bin_means)...))) |>
    @where(__, .!ismissing.(:time_bin_mean)) |>
    groupby(__, [:condition, :target_time_label, :switch_index]) |>
    filteringmap(__, folder = foldl, desc = nothing, :time =>
        map(binmean -> (binmean => (row ->
            abs(row.time_bin_mean - binmean) < binwidth/2)), bin_means),
        function(sdf, time)
            result = @combine(groupby(sdf, [:sid, :exp_id]),
                mean = sum(:perf .== "hit") / sum(:perf .∈ Ref(Set(["hit", "miss"]))),
                weight = sum(:perf .∈ Ref(Set(["hit", "miss"]))),
            )
            allowmissing!(result, :mean)
            for (sid, exp_id) in setdiff(allids, getids(sdf))
                result = push!!(result, (
                    sid = sid, exp_id = exp_id, mean = missing, weight = 0))
            end
            result
        end
    ) |>
    sort!(__, :time) |>
    @transform(__, mean = extend(ifelse.(iszero.(:weight), missing, :mean))) |>
    groupby(__, [:condition, :time, :target_time_label, :sid, :exp_id, :switch_index]) |>
    @combine(__,
        mean = all(ismissing, :mean) ? missing : mean(skipmissing(:mean)),
        weight = sum(:weight)
    ) |>
    groupby(__, [:condition, :time, :target_time_label]) |>
    combine(function(sdf)
        filt = filter(x -> !ismissing(x.mean), sdf)
        μ, lower, upper = if isempty(filt)
            missing, missing, missing
        else
            μ, lower, upper = confint(bootstrap(df -> mean(df.mean), filt,
                BasicSampling(10_000)), BasicConfInt(0.95)
            )[1]
        end
        DataFrame(mean = μ, lower = lower, upper = upper, weight = sum(filt.weight .> 0))
    end, __) |>
    @transform(__, time = Array(:time))

CSV.write(joinpath(processed_datadir("analyses", "hit_by_switch.csv")), hit_by_switch)

run(`Rscript $(joinpath(scriptsdir("R"), "nearfar_behavior.R"))`)

file = joinpath(processed_datadir("analyses"), "nearfar_behavior_coefs.csv")
pldata = CSV.read(file, DataFrame)
barwidth = 14
ytitle = ["Slope of Hit Rate by Time"]
pl = @_ pldata |>
    rename(__, :r_med => :mean, :r_05 => :lower, :r_95 => :upper) |>
    @transform(__,
        condition_time = :comparison,
        condition = string.(getindex.(split.(:comparison, "_"),1)),
        target_time_label = string.(getindex.(split.(:comparison, "_"),2)),
    ) |>
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
        y = {:mean, title = ytitle}
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
        y = {datum = 0},
        text = {value = "Early"},
    ) +
    @vlplot({:text, angle = -90, fontSize = 9, align = "left", baseline = "top", dx = 0, dy = barwidth+2},
        transform = [{filter = "datum.target_time_label == 'late' && datum.condition == 'global'"}],
        # x = {datum = "spatial"}, y = {datum = },
        x = {:condition, axis = {title = ""}},
        y = {datum = 0},
        text = {value = "Late"},
    );
plotfile = joinpath(dir, "fig4b.svg")
pl |> save(plotfile)
addpatterns(plotfile, inpatterns, size = 10)

function fitangle(x,y)
    if length(x) > 1
        model = glm(@formula(y ~ x), DataFrame(x = x, y = y), Bernoulli(), LogitLink())
        coef(model)[2]
    else
        missing
    end
end
target_angles = @_ hit_by_switch |>
    @where(__, :perf .∈ Ref(Set(["hit", "miss"]))) |>
    groupby(__, [:condition, :target_time_label]) |>
    @combine(__, angle = fitangle(:direction_timing, :sbj_answer))

pl = @_ target_angles |>
    groupby(__, [:condition, :target_time_label]) |>
    @combine(__, angle = quantile(collect(skipmissing(:angle)), 0.6))

pl = @_ target_angles |>
    @where(__, .!ismissing.(:angle)) |>
    @vlplot(:bar,
        column = :target_time_label,
        x = :condition,
        y = {:angle, aggregate = :mean, type = :quantitative}
    );
pl |> save(joinpath(dir, "supplement", "angles.svg"))


# NOTE: unused attempt to use merve analysis data
file1 = joinpath(processed_datadir("behavioral", "merve_summaries", "export_switch_ind_data_1.csv"))
file2 = joinpath(processed_datadir("behavioral", "merve_summaries", "export_switch_ind_data_2.csv"))
switchdata = append!(cols = :union,
    @_(CSV.read(file1, DataFrame) |>
        transform!(__, :block_type => ByRow(x -> ascondition[x]) => :condition) |>
        insertcols!(__, :target_time_label => "early")),
    @_(CSV.read(file2, DataFrame) |>
        transform!(__, :block_type => ByRow(x -> ascondition[x]) => :condition) |>
        insertcols!(__, :target_time_label => "late"))) |>
    x -> rename!(x, :sbj_id => :sid)

nbins = 30
binwidth = 1.5
qs = quantile(skipmissing(switchdata.direction_timing), range(0, 1, length = nbins+1))
bin_means = (qs[1:(end-1)] + qs[2:end])/2
getids(df) = @_ df |> groupby(__, [:sid, :exp_id]) |>
    combine(first, __) |> zip(__.sid, __.exp_id)
allids = getids(switchdata)
target_timeline = @_ switchdata |>
    # @where(__, :sid .∉ bad_sids) |>
    @transform(__, time_bin = cut(:direction_timing, nbins+1, allowempty = true)) |>
    @transform(__,
        time_bin_mean = Array(recode(:time_bin, (levels(:time_bin) .=> bin_means)...))) |>
    @where(__, .!ismissing.(:time_bin_mean))


# NOTE: using my analysis from raw data
target_timeline = @_ hit_by_switch |>
    groupby(__, [:condition, :target_time_label]) |>
    filteringmap(__, folder = foldl, desc = nothing, :time =>
        map(binmean -> (binmean => (row ->
            abs(row.time_bin_mean - binmean) < binwidth/2)), bin_means),
        function(sdf, time)
            result = @combine(groupby(sdf, [:sid, :exp_id]),
                mean = sum(:perf .== "hit") / sum(:perf .∈ Ref(Set(["hit", "miss"]))),
                weight = sum(:perf .∈ Ref(Set(["hit", "miss"]))),
            )
            allowmissing!(result, :mean)
            for (sid, exp_id) in setdiff(allids, getids(sdf))
                result = push!!(result, (
                    sid = sid, exp_id = exp_id, mean = missing, weight = 0))
            end
            result
        end
    ) |>
    sort!(__, :time) |>
    @transform(__, mean = extend(ifelse.(iszero.(:weight), missing, :mean))) |>
    groupby(__, [:condition, :time, :target_time_label, :sid, :exp_id]) |>
    @combine(__,
        mean = all(ismissing, :mean) ? missing : mean(skipmissing(:mean)),
        weight = sum(:weight)
    ) |>
    groupby(__, [:condition, :time, :target_time_label]) |>
    combine(function(sdf)
        filt = filter(x -> !ismissing(x.mean), sdf)
        μ, lower, upper = if isempty(filt)
            missing, missing, missing
        else
            μ, lower, upper = confint(bootstrap(df -> mean(df.mean), filt,
                BasicSampling(10_000)), BasicConfInt(0.95)
            )[1]
        end
        DataFrame(mean = μ, lower = lower, upper = upper, weight = sum(filt.weight .> 0))
    end, __) |>
    @transform(__, time = Array(:time))

last_time = maximum(rawdata.direction_timing)

pl = @_ target_timeline |>
    @vlplot(
        config = {legend = {disable = true}},
        transform = [{calculate = "upper(slice(datum.condition,0,1)) + slice(datum.condition,1)",
                        as = :condition}],
        spacing = 1,
        facet = {
            column = {
                field = :target_time_label, type = :ordinal, title = nothing,
                sort = ["early", "late"],
                header = {labelFontWeight = "bold"}
            }
        }
    ) +
    (
        @vlplot(width = 80, autosize = "fit", height = 130,
            color = {:condition, scale = {range = "#".*hex.(colors)}}) +
        @vlplot({:trail, clip = true},
            transform = [{filter = "datum.time < $(last_time) || datum.target_time == 'early'"}],
            x = {:time, type = :quantitative, scale = {domain = [0, last_time+0.25]},
                title = ["Time after", "Switch (s)"]},
            y = {:mean, type = :quantitative, scale = {domain = [0.5, 1]}, title = "Hit Rate"},
            size = {:weight, type = :quantitative, scale = {range = [0, 2]}},
        ) +
        # @vlplot({:point, filled = true, size = 15}, x = :time, y = :mean) +
        @vlplot({:errorband, clip = true},
            transform = [{filter = "datum.time < $(last_time) || datum.target_time == 'early'"}],
            x = {:time, type = :quantitative, scale = {domain = [0, last_time+0.25]}},
            y = {:upper, type = :quantitative, title = "", scale = {domain = [0.5, 1]}}, y2 = :lower,
            # opacity = :weight,
            color = :condition,
        ) +
        # @vlplot({:text, align = :left, dx = 5},
        #     transform = [
        #         {filter = "datum.time > 1 && datum.time < 1.1 && datum.target_time == 'late'"},
        #     ],
        #     x = {datum = 1.2},
        #     y = {:mean, aggregate = :mean, type = :quantitative},
        #     color = :condition,
        #     text = {:condition, }
        # ) +
        (
            @vlplot(data = {values = [{}]}) +
            # @vlplot({:rule, strokeDash = [4 4], size = 1},
            #     x = {datum = nearsplit},
            #     color = {value = "black"}
            # ) +
            @vlplot({:text, fontSize = 9, align = :right, dx = 2, baseline = "bottom"},
                x = {datum = 1.5},
                y = {datum = 0.5},
                text = {value = "Far"},
                color = {value = "#"*hex(darkgray)}
            ) +
            @vlplot({:text, fontSize = 9, align = :left, dx = 2, baseline = "bottom"},
                x = {datum = 0},
                y = {datum = 0.5},
                text = {value = "Near"},
                color = {value = "#"*hex(darkgray)}
            )
        )
    );
pl |> save(joinpath(dir, "fig4a.svg"))

# Find hyperparameters (λ and winlen)
# =================================================================

nbins = 10
switchbreaks = @_ GermanTrack.load_stimulus_metadata().switch_distance |>
    skipmissing |>
    quantile(__, range(0,1,length = nbins+1)[2:(end-1)])

file = joinpath(processed_datadir("analyses"), "nearfar-hyperparams.json")
GermanTrack.@cache_results file fold_map hyperparams begin
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
        addfold!(__, 10, :sid, rng = stableRNG(2019_11_18, :nearfar_hyper_folds)) |>
        groupby(__, [:winstart, :winlen, :condition, :switch_break]) |>
        filteringmap(__, folder = foldxt, desc = "Evaluating hyperparameters...",
            :cross_fold => 1:10,
            function(sdf, fold)
                test, model = traintest(sdf, fold, y = :target_switch_label, weight = :weight,
                    selector = m -> AllSeg(), λ = lambdas)

                # remove the channels: they are redundant, and bloat memory
                test[:, Not(r"channel")]
            end) |>
        deletecols!(__, :windows)

    fold_map = @_ resultdf |>
        groupby(__, :sid) |> combine(__, :fold => first => :fold) |>
        Dict(row.sid => row.fold for row in eachrow(__))

    # ISSUE: were selecting using the test data, but we should select
    # using the train data (given how close folds are to one another,
    # I don't think this is going to change much, but we should *DO* it)
    hyperparamsdf =
        filteringmap(resultdf, desc = nothing, folder = foldxt,
            :fold => cross_folds(1:10),
            function (sdf, fold)
                λ = @_ sdf.λ |> unique |> __[argmin(abs.(__ .- 0.13))]

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

    hyperparams = Dict(row.fold => NamedTuple(row[Not(:fold)])
        for row in eachrow(hyperparamsdf))

    @info "Saving plots to $(joinpath(dir, "supplement"))"
end

# Plot near/far across early/late (Fig 4c)
# =================================================================

# Classification accuracy
# -----------------------------------------------------------------

file = joinpath(cache_dir("features"), "nearfar-target.json")
GermanTrack.@cache_results file resultdf begin
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    windims = @_ getindex.(values(hyperparams), [:winstart :winlen]) |>
        mapslices(Tuple, __, dims = 2) |>
        unique

    classdf = @_ events |> filter(ishit(_) == "hit", __) |>
        transform!(__, :sid => ByRow(x -> fold_map[x]) => :fold) |>
        transform!(__, [:fold, :switch_distance] =>
            ByRow((f,d) -> ismissing(d) ? missing :
                (d >= switchbreaks[hyperparams[f].switch_break] ? "near" : "far")) => :target_switch_label) |>
        groupby(__, [:sid, :fold, :condition, :target_time_label, :target_switch_label]) |>
        filteringmap(__, desc = "Computing features...",
            :window => [windowtarget(start = start, len = len) for (start,len) in windims],
            compute_powerbin_features(_1, subjects, _2)) |>
        deletecols!(__, :window)

    resultdf = @_ classdf |>
        addfold!(__, 10, :sid, rng = stableRNG(2019_11_18, :)) |>
        transform!(__, :sid => ByRow(x -> fold_map[x]) => :fold) |>
        groupby(__, [:condition, :target_time_label]) |>
        filteringmap(__, folder = foldl, desc = "Classifying target proximity...",
            :cross_fold => 1:10,
            :modeltype => ["full", "null"],
            function(sdf, fold, modeltype)
                selector = modeltype == "null" ? m -> NullSelect() : hyperparams[fold][:λ]
                lens = hyperparams[fold][:winlen] #|> GermanTrack.spread(0.5, n_winlens)
                start = hyperparams[fold][:winstart]

                sdf = filter(x -> x.winlen ∈ lens && x.winstart == start, sdf)
                test, model = traintest(sdf, fold, y = :target_switch_label,
                    selector = selector, weight = :weight)

                test[:, Not(r"channel")]
            end)

    # GermanTrack.@store_cache file resultdf
end

# Plot data
# -----------------------------------------------------------------

# classmeans = @_ hyper_resultdf |>
#     filter(_.switch_break == 3, __) |>
#     filter(_.λ ∈ [1.0, λ_map[_.fold]], __) |>
classmeans = @_ resultdf |>
    groupby(__, [:winstart, :winlen, :sid, :modeltype, :fold, :condition, :target_time_label]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :mean)


classmeans_sum = @_ classmeans |>
    groupby(__, [:winstart, :sid, :modeltype, :fold, :condition, :target_time_label]) |>
    @combine(__, mean = maximum(:mean)) |>
    groupby(__, [:sid, :modeltype, :fold, :condition, :target_time_label]) |>
    @combine(__, mean = mean(:mean))

statdata = @_ classmeans_sum |>
    groupby(__, [:condition, :sid, :target_time_label]) |>
    @transform(__,
        logitnullmean = logit(shrink(only(:mean[:modeltype .== "null"]))),
        logitmean     = logit.(shrink.(:mean)),
    ) |>
    @transform(__,
        nullodds = exp.(:logitnullmean),
        odds     = exp.(:logitmean)
    ) |>
    @where(__, :modeltype .== "full")

pl = statdata |> @vlplot(:point,
    column = :condition,
    color = :target_time_label,
    x     = {:nullodds, scale = {type = :log, base = 10}, axis = {grid = false}},
    y     = {:odds, scale = {type = :log, base = 10}, axis = {grid = false}},
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
        height = 150, width = 160,
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

# Combine early/late plots
# -----------------------------------------------------------------

GermanTrack.@usepython

svg = pyimport("svgutils").compose

background_file = tempname()

background = pyimport("svgutils").transform.fromstring("""
    <svg>
        <rect width="100%" height="100%" fill="white"/>
    </svg>
""").save(background_file)

for (suffix, file) in [
    ("behavior_timeline", "fig4a.svg"),
    ("behavior_slopes", "fig4b.svg"),
    ("neural", "fig4c.svg")]
    filereplace(joinpath(dir, file), r"\bclip([0-9]+)\b" =>
        SubstitutionString("clip\\1_$suffix"))
end

fig = svg.Figure("89mm", "240mm", # "240mm",
    svg.SVG(background_file),
    svg.Panel(
        svg.SVG(joinpath(dir, "fig4a.svg")).move(0,15),
        svg.Text("A", 2, 10, size = 12, weight="bold", font = "Helvetica"),
        svg.SVG(joinpath(plotsdir("icons"), "behavior.svg")).
            scale(0.1).move(115,30),
        svg.SVG(joinpath(plotsdir("icons"), "behavior.svg")).
            scale(0.1).move(220,30)
    ).move(0, 0),
    svg.Panel(
        svg.SVG(joinpath(dir, "fig4b.svg")).move(0,15),
        svg.Text("B", 2, 10, size = 12, weight="bold", font = "Helvetica"),
        svg.SVG(joinpath(plotsdir("icons"), "behavior.svg")).
            scale(0.1).move(220,10)
    ).move(0, 250),
    svg.Panel(
        svg.SVG(joinpath(dir, "fig4c.svg")).move(0,15),
        svg.Text("C", 2, 10, size = 12, weight = "bold", font = "Helvetica"),
        svg.SVG(joinpath(plotsdir("icons"), "eeg.svg")).
            scale(0.1).move(220,25)
    ).move(0, 450),
).scale(1.333).save(joinpath(plotsdir("figures"), "fig4.svg"))

