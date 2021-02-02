# Setup
# =================================================================

using DrWatson
@quickactivate("german_track")

using GermanTrack, DataFrames, Statistics, Dates, Underscores, Random, Printf,
    ProgressMeter, VegaLite, FileIO, StatsBase, BangBang, Transducers, Infiltrator, Peaks,
    StatsFuns, Distributions, DSP, DataStructures, Colors, Bootstrap, CSV, EEGCoding,
    JSON3, DataFramesMeta, Lasso
wmean = GermanTrack.wmean
n_winlens = 6

dir = mkpath(joinpath(plotsdir(), "figure2_parts"))

using GermanTrack: colors, neutral, patterns

# Fig 2A: behavioarl hit rate
# =================================================================

ascondition = Dict(
    "test" => "global",
    "feature" => "spatial",
    "object" => "object"
)

file = joinpath(raw_datadir("behavioral", "export_ind_data.csv"))
rawdata = @_ CSV.read(file, DataFrame) |>
    transform!(__, :block_type => ByRow(x -> ascondition[x]) => :condition)

meansraw = @_ rawdata |>
    groupby(__, [:sid, :condition, :exp_id]) |>
    @combine(__,
        hr = sum(:perf .== "hit") / sum(:perf .∈ Ref(Set(["hit", "miss"]))),
        fr = sum(:perf .== "false") / sum(:perf .∈ Ref(Set(["false", "reject"]))),
    )

bad_sids = @_ meansraw |>
    @where(__, :condition .== "global") |>
    @where(__, (:hr .<= :fr) .| (:fr .>= 1)) |> __.sid |> Ref |> Set

meansclean = @_ meansraw |>
    @where(__, :sid .∉ bad_sids) |>
    stack(__, [:hr, :fr], [:sid, :condition, :exp_id], variable_name = :type, value_name = :prop)


CSV.write(joinpath(processed_datadir("analyses"), "behavioral_condition.csv"), meansclean)

run(`Rscript $(joinpath(scriptsdir("R"), "condition_behavior.R"))`)

file = joinpath(processed_datadir("analyses"), "behavioral_condition_coefs.csv")
means = @_ CSV.read(file, DataFrame) |>
    rename(__, :propr_med => :prop, :propr_05 => :lower, :propr_95 => :upper) |>
    @transform(__, condition = categorical(:condition,
        levels = ["global", "spatial", "object"], ordered = true)) |>
    sort!(__, :condition)

barwidth = 20
pl = means |> @vlplot(
        width = 190, height = 130,
        # width = {step = 50},
        x = {:condition,
            type = :nominal,
            sort = ["global", "spatial", "object"],
            axis = {title = "", labelAngle = 0,
            labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}, },
        config = {
            bar = {discreteBandSize = barwidth},
            axis = {labelFont = "Helvetica", titleFont = "Helvetica"},
            legend = {disable = true, labelFont = "Helvetica", titleFont = "Helvetica"},
            header = {labelFont = "Helvetica", titleFont = "Helvetica"},
            mark = {font = "Helvetica"},
            text = {font = "Helvetica"},
            title = {font = "Helvetica", subtitleFont = "Helvetica"}
        }) +
    @vlplot({:bar, xOffset = -(barwidth/2)},
        transform = [{filter = "datum.type == 'hr'"}],
        y = {:prop, type = :quantitative, aggregate = :mean,
                scale = {domain = [0, 1]}, title = "Response Rate"},
        color = {:condition, scale = {range = "#".*hex.(colors)}}) +
    @vlplot({:line, xOffset = -(barwidth/2), size = 1},
        transform = [{filter = "datum.type == 'hr'"}],
        y = {:prop, type = :quantitative, aggregate = :mean,
                scale = {domain = [0, 1]}, title = "Response Rate"},
        color = {value = "black"}) +
    @vlplot({:bar, xOffset = (barwidth/2)},
        transform = [{filter = "datum.type == 'fr'"}],
        y = {:prop, type = :quantitative, aggregate = :mean},
        color = {value = "#"*hex(neutral)}) +
    @vlplot({:line, xOffset = (barwidth/2), size = 1},
        transform = [{filter = "datum.type == 'fr'"}],
        y = {:prop, type = :quantitative, aggregate = :mean,
                scale = {domain = [0, 1]}, title = "Response Rate"},
        color = {value = "black"}) +
    @vlplot({:rule, xOffset = -(barwidth/2)},
        transform = [{filter = "datum.type == 'hr'"}],
        color = {value = "black"},
        y = {:lower, title = ""}, y2 = :upper
    ) +
    @vlplot({:rule, xOffset = (barwidth/2)},
        transform = [{filter = "datum.type == 'fr'"}],
        color = {value = "black"},
        y = {:lower, title = ""}, y2 = :upper
    ) +
    @vlplot({:text, angle = -90, fontSize = 9, align = "right", baseline = "bottom", dx = 0, dy = -barwidth-2},
        transform = [{filter = "datum.condition == 'global' && datum.type == 'hr'"}],
        # x = {datum = "spatial"}, y = {datum = 0.},
        y = {:prop, aggregate = :mean, type = :quantitative},
        text = {value = "Hits"},
    ) +
    @vlplot({:text, angle = -90, fontSize = 9, align = "left", baseline = "top", dx = 0, dy = barwidth+2},
        transform = [{filter = "datum.condition == 'global' && datum.type == 'fr'"}],
        # x = {datum = "spatial"}, y = {datum = },
        y = {datum = 0},
        text = {value = "False Positives"},
    );
pl |> save(joinpath(dir, "fig2a.svg"))

# Presentation
# -----------------------------------------------------------------

barwidth = 13
pl = means |> @vlplot(
        width = 145, height = 75,
        # width = {step = 50},
        x = {:condition,
            type = :nominal,
            sort = ["global", "spatial", "object"],
            axis = {title = "", labelAngle = 0,
            labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}, },
        config = {
            bar = {discreteBandSize = barwidth},
            axis = {labelFont = "Helvetica", titleFont = "Helvetica"},
            legend = {disable = true, labelFont = "Helvetica", titleFont = "Helvetica"},
            header = {labelFont = "Helvetica", titleFont = "Helvetica"},
            mark = {font = "Helvetica"},
            text = {font = "Helvetica"},
            title = {font = "Helvetica", subtitleFont = "Helvetica"}
        }) +
    @vlplot({:bar, xOffset = -(barwidth/2)},
        transform = [{filter = "datum.type == 'hr'"}],
        y = {:prop, type = :quantitative, aggregate = :mean,
                scale = {domain = [0, 1]}, title = "Response Rate"},
        color = {:condition, scale = {range = "#".*hex.(colors)}}) +
    @vlplot({:line, xOffset = -(barwidth/2), size = 1},
        transform = [{filter = "datum.type == 'hr'"}],
        y = {:prop, type = :quantitative, aggregate = :mean,
                scale = {domain = [0, 1]}, title = "Response Rate"},
        color = {value = "black"}) +
    @vlplot({:bar, xOffset = (barwidth/2)},
        transform = [{filter = "datum.type == 'fr'"}],
        y = {:prop, type = :quantitative, aggregate = :mean},
        color = {value = "#"*hex(neutral)}) +
    @vlplot({:line, xOffset = (barwidth/2), size = 1},
        transform = [{filter = "datum.type == 'fr'"}],
        y = {:prop, type = :quantitative, aggregate = :mean,
                scale = {domain = [0, 1]}, title = "Response Rate"},
        color = {value = "black"}) +
    @vlplot({:rule, xOffset = -(barwidth/2)},
        transform = [{filter = "datum.type == 'hr'"}],
        color = {value = "black"},
        y = {:lower, title = ""}, y2 = :upper
    ) +
    @vlplot({:rule, xOffset = (barwidth/2)},
        transform = [{filter = "datum.type == 'fr'"}],
        color = {value = "black"},
        y = {:lower, title = ""}, y2 = :upper
    ) +
    @vlplot({:text, angle = -90, fontSize = 9, align = "right", baseline = "bottom", dx = 0, dy = -barwidth-1},
        transform = [{filter = "datum.condition == 'global' && datum.type == 'hr'"}],
        # x = {datum = "spatial"}, y = {datum = 0.},
        y = {:prop, aggregate = :mean, type = :quantitative},
        text = {value = "Hits"},
    ) +
    @vlplot({:text, angle = -90, fontSize = 9, align = "left", baseline = "top", dx = 0, dy = barwidth+1},
        transform = [{filter = "datum.condition == 'global' && datum.type == 'fr'"}],
        # x = {datum = "spatial"}, y = {datum = },
        y = {datum = 0},
        text = {value = "False Positives"},
    );
pl |> save(joinpath(dir, "present", "fig2a.svg"))


# Find best λs
# =================================================================

file = joinpath(processed_datadir("analyses"), "condition_lambdas.json")
GermanTrack.@cache_results file fold_map hyperparams begin
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    classdf = @_ events |>
        filter(ishit(_) == "hit", __) |>
        groupby(__, [:sid, :condition]) |>
        filteringmap(__, desc = "Computing features...",
            :window => [windowtarget(len = len, start = start)
                for len in 2.0 .^ range(-1, 1, length = 10),
                    start in [0; 2.0 .^ range(-2, 2, length = 10)]],
            compute_powerbin_features(_1, subjects, _2)) |>
        deletecols!(__, :window)

    resultdf = @_ classdf |>
        addfold!(__, 10, :sid, rng = stableRNG(2019_11_18, :condition_lambda_folds)) |>
        groupby(__, [:winstart, :winlen]) |>
        filteringmap(__, desc = "Evaluating lambdas...", folder = foldxt,
            :cross_fold => 1:10,
            :comparison => (
                "global-v-object"  => x -> x.condition ∈ ["global", "object"],
                "global-v-spatial" => x -> x.condition ∈ ["global", "spatial"],
                "object-v-spatial" => x -> x.condition ∈ ["object", "spatial"],
            ),
            function(sdf, fold, comparison)
                test, model = traintest(sdf, fold, y = :condition)
                test.nzero = sum(!iszero, coef(model, MinAICc()))

                test
            end)

    fold_map = @_ resultdf |>
        groupby(__, :sid) |> combine(__, :fold => first => :fold) |>
        Dict(row.sid => row.fold for row in eachrow(__))

    win_means = @_ resultdf |>
        groupby(__, [:comparison, :winlen, :winstart, :sid, :fold]) |>
        @combine(__, mean = GermanTrack.wmean(:correct, :weight)) |>
        groupby(__, [:winlen, :winstart]) |>
        filteringmap(__, desc = nothing,
            :fold => cross_folds(1:10),
            (sdf, fold) -> DataFrame(mean = mean(sdf.mean)))

    @_ win_means |>
        @vlplot(:line,
            # column = :comparison,
            x = :winstart, y = :mean,
            color = :winlen
        ) |> save(joinpath(dir, "supplement", "window_means.svg"))

    hyperparams = @_ win_means |>
        groupby(__, :fold) |>
        @combine(__,
            best = maximum(:mean),
            winlen = :winlen[argmax(:mean)],
            winstart = :winstart[argmax(:mean)]
        ) |>
        Dict(row.fold => (;row[Not(:fold)]...) for row in eachrow(__))

    @info "Saving plots to $(joinpath(dir, "supplement"))"
end

# Compute condition categorization, and several baselines (sanity checks)
# =================================================================

file = joinpath(processed_datadir("analyses"), "condition-and-baseline.json")
GermanTrack.@cache_results file predictbasedf begin
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    windowtypes = [
        "target"    => (;kwds...) -> windowtarget(name = "target"; kwds...)
        "rndbefore" => (;kwds...) -> windowbase_bytarget(>; name = "rndbefore",
            mindist = 0.5, minlength = 0.5, onempty = missing, kwds...)
    ]

    start_len = unique((x.winstart, x.winlen) for x in values(hyperparams))

    classdf = @_ events |>
        transform!(__, AsTable(:) => ByRow(ishit) => :hittype) |>
        transform!(__, :sid => ByRow(x -> fold_map[x]) => :fold) |>
        groupby(__, [:sid, :condition, :hittype]) |>
        filteringmap(__, desc = "Computing features...",
            :window => [winfn(start = start, len = len)
                for (name, winfn) in windowtypes
                for (start, len) in start_len],
            compute_powerbin_features(_1, subjects, _2)) |>
        deletecols!(__, :window)

    modeltypes = (
        "full" => x -> x.windowtype == "target",
        "null" => x -> x.windowtype == "target",
        "random-labels" => x -> x.windowtype == "target",
        "random-window-before" => x -> x.windowtype == "rndbefore",
        "random-trialtype" => x -> x.windowtype == "target"
    )

    modelshufflers = Dict(
        "random-labels" => (df -> @_ df |>
            groupby(__, [:sid, :winlen, :windowtype, :hittype]) |>
            transform(__, :condition => shuffle => :condition)),
        "random-trialtype" => (df -> @_ df |>
            groupby(__, [:sid, :condition, :winlen, :windowtype]) |>
            transform(__, :hittype => shuffle => :hittype))
    )

    predictbasedf = @_ classdf |>
        transform!(__, :sid => ByRow(sid -> fold_map[sid]) => :fold) |>
        groupby(__, :hittype) |>
        filteringmap(__, desc = "Classifying conditions...",
            :cross_fold => 1:10,
            :comparison => (
                "global-v-object"  => x -> x.condition ∈ ["global", "object"],
                "global-v-spatial" => x -> x.condition ∈ ["global", "spatial"],
                "object-v-spatial" => x -> x.condition ∈ ["object", "spatial"],
            ),
            :modeltype => modeltypes,
            function (sdf, fold, comparison, modeltype)
                selector = modeltype == "null" ? NullSelect() : MinAICc()
                sdf = get(modelshufflers, modeltype, identity)(sdf)

                win = hyperparams[fold]
                sdf = filter(x -> x.winstart == win.winstart && x.winlen == win.winlen, sdf)
                conds = split(comparison, "-v-")

                test, model = traintest(sdf, fold, y = :condition, selector = selector)
                test.nzero = sum(!iszero, coef(model, selector))

                test
            end)
end

# Main EEG results (Figure 2B)
# =================================================================

predictmeans = @_ predictbasedf |>
    filter(_.modeltype ∈ ["null", "full"] && _.hittype == "hit", __) |>
    groupby(__, [:sid, :comparison, :modeltype, :winlen, :hittype]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :correct) |>
    groupby(__, [:sid, :comparison, :modeltype, :hittype]) |>
    combine(__,
        :correct => mean => :mean,
        :correct => logit ∘ shrinktowards(0.5, by=0.01) ∘ mean => :logitcorrect)

nullmeans = @_ predictmeans |>
    filter(_.modeltype == "null", __) |>
    deletecols!(__, [:logitcorrect, :modeltype]) |>
    rename!(__, :mean => :nullmean)

statdata = @_ predictmeans |>
    filter(_.modeltype == "full", __) |>
    innerjoin(__, nullmeans, on = [:sid, :comparison, :hittype]) |>
    transform!(__, :mean => ByRow(shrinktowards(0.5, by = 0.01)) => :shrinkmean) |>
    transform!(__, :mean => ByRow(logit ∘ shrinktowards(0.5, by = 0.01)) => :logitmean) |>
    transform!(__, :nullmean => ByRow(logit ∘ shrinktowards(0.5, by = 0.01)) => :logitnullmean)

file = joinpath(processed_datadir("analyses"), "eeg_condition.csv")
CSV.write(file, statdata)
run(`Rscript $(joinpath(scriptsdir("R"), "condition_eeg.R"))`)

compnames = OrderedDict(
    "global_v_object"  => "Global vs.\n Object",
    "global_v_spatial" => "Global vs.\n Spatial",
    "object_v_spatial" => "Object vs.\n Spatial")

statfile = joinpath(processed_datadir("analyses"), "eeg_condition_coefs.csv")
plotdata = @_ CSV.read(statfile, DataFrame) |>
    rename(__, :propr_med => :mean, :propr_05 => :lower, :propr_95 => :upper) |>
    @transform(__, compname = map(x -> compnames[x], :comparison))
nullmean = logistic(mean(statdata.logitnullmean))

ytitle = "Condition Classification"
barwidth = 25
plhit = @_ plotdata |>
    @vlplot(
        # facet = { column = { field = :hittype, type = :nominal} },
        width = 180, height = 130,
        config = {
            bar = {discreteBandSize = barwidth},
            axis = {labelFont = "Helvetica", titleFont = "Helvetica"},
            legend = {disable = true, labelFont = "Helvetica", titleFont = "Helvetica"},
            header = {labelFont = "Helvetica", titleFont = "Helvetica"},
            mark = {font = "Helvetica"},
            text = {font = "Helvetica"},
            title = {font = "Helvetica", subtitleFont = "Helvetica"}
        }
    ) + (
    @vlplot(x = {:compname, axis = {
            labelAngle = 0,
            title = "",
            labelExpr = "split(datum.label, '\\n')"}},
        color = {
            :compname, title = nothing,
            scale = {range = ["url(#mix1_2)", "url(#mix1_3)", "url(#mix2_3)"]}}) +
    @vlplot({:bar},
        y = {:mean,
            scale = {domain = [0.5, 1]},
            title = ytitle}) +
    @vlplot({:rule},
        color = {value = "black"},
        y2 = :upper,
        y = {:lower,
            scale = {domain = [0.5, 1]},
            title = ytitle})
    ) +
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:rule, strokeDash = [4 4], size = 2},
            y = {datum = nullmean},
            color = {value = "black"}) +
        @vlplot({:text, align = "left", dx = 15, dy = 0, baseline = "line-bottom", fontSize = 9},
            y = {datum = nullmean},
            x = {datum = "Object vs.\n Spatial"},
            text = {value = ["Null Model", "Accuracy"]}
        )
    );
plotfile = joinpath(dir, "fig2b.svg")
plhit |> save(plotfile)
addpatterns(plotfile, patterns, size = 10)

# Presentation version
# -----------------------------------------------------------------

ytitle = "Accuracy"
barwidth = 25
plhit = @_ plotdata |>
    @vlplot(
        # facet = { column = { field = :hittype, type = :nominal} },
        width = 145, height = 75,
        config = {
            bar = {discreteBandSize = barwidth},
            axis = {labelFont = "Helvetica", titleFont = "Helvetica"},
            legend = {disable = true, labelFont = "Helvetica", titleFont = "Helvetica"},
            header = {labelFont = "Helvetica", titleFont = "Helvetica"},
            mark = {font = "Helvetica"},
            text = {font = "Helvetica"},
            title = {font = "Helvetica", subtitleFont = "Helvetica"}
        }
    ) + (
    @vlplot(x = {:compname, axis = {
            labelAngle = 0,
            title = "",
            labelExpr = "split(datum.label, '\\n')"}},
        color = {
            :compname, title = nothing,
            scale = {range = ["url(#mix1_2)", "url(#mix1_3)", "url(#mix2_3)"]}}) +
    @vlplot({:bar},
        y = {:mean,
            scale = {domain = [0.5, 1]},
            title = ytitle}) +
    @vlplot({:rule},
        color = {value = "black"},
        y2 = :upper,
        y = {:lower,
            scale = {domain = [0.5, 1]},
            title = ytitle})
    );
plotfile = joinpath(dir, "present", "fig2b.svg")
plhit |> save(plotfile)
addpatterns(plotfile, patterns, size = 10)

# Early/late condition classifiers
# =================================================================

subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

start_len = unique((x[:winstart], x[:winlen]) for x in values(hyperparams))

classdf = @_ events |>
    transform!(__, AsTable(:) => ByRow(ishit) => :hittype) |>
    @where(__, :hittype .== "hit") |>
    transform!(__, :sid => ByRow(x -> fold_map[x]) => :fold) |>
    groupby(__, [:sid, :condition, :hittype, :target_time_label]) |>
    filteringmap(__, desc = "Computing features...",
        :window => [windowtarget(start = start, len = len)
            for (start, len) in start_len],
        compute_powerbin_features(_1, subjects, _2)) |>
    deletecols!(__, :window)

cond_earlylate_df = @_ classdf |>
    transform!(__, :sid => ByRow(sid -> fold_map[sid]) => :fold) |>
    groupby(__, :target_time_label) |>
    filteringmap(__, desc = "Classifying conditions...", folder = foldl,
        :cross_fold => 1:10,
        :comparison => (
            "global-v-object"  => x -> x.condition ∈ ["global", "object"],
            "global-v-spatial" => x -> x.condition ∈ ["global", "spatial"],
            "object-v-spatial" => x -> x.condition ∈ ["object", "spatial"],
        ),
        :modeltype => ["full", "null"],
        function (sdf, fold, comparison, modeltype)
            selector = modeltype == "null" ? x -> NullSelect() : x -> MinAICc()

            win = hyperparams[fold]
            sdf = filter(x -> x.winstart == win[:winstart] && x.winlen == win[:winlen], sdf)
            conds = split(comparison, "-v-")

            @infiltrate
            test, model = traintest(sdf, fold, y = :condition, selector = selector, weight = :weight)
            test.nzero = sum(!iszero, coef(model, selector(model)))

            test
        end)


predictmeans = @_ cond_earlylate_df |>
    filter(_.modeltype ∈ ["null", "full"] && _.hittype == "hit", __) |>
    groupby(__, [:sid, :comparison, :modeltype, :winlen, :winstart, :target_time_label]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :correct) |>
    groupby(__, [:sid, :comparison, :modeltype, :target_time_label]) |>
    combine(__,
        :correct => mean => :mean,
        :correct => logit ∘ shrinktowards(0.5, by=0.01) ∘ mean => :logitcorrect)

nullmeans = @_ predictmeans |>
    filter(_.modeltype == "null", __) |>
    deletecols!(__, [:logitcorrect, :modeltype]) |>
    rename!(__, :mean => :nullmean)

statdata = @_ predictmeans |>
    filter(_.modeltype == "full", __) |>
    innerjoin(__, nullmeans, on = [:sid, :comparison, :target_time_label]) |>
    transform!(__, :mean => ByRow(shrinktowards(0.5, by = 0.01)) => :shrinkmean) |>
    transform!(__, :mean => ByRow(logit ∘ shrinktowards(0.5, by = 0.01)) => :logitmean) |>
    transform!(__, :nullmean => ByRow(logit ∘ shrinktowards(0.5, by = 0.01)) => :logitnullmean)


compnames = OrderedDict(
    "global-v-object"  => "Global vs.\n Object",
    "global-v-spatial" => "Global vs.\n Spatial",
    "object-v-spatial" => "Object vs.\n Spatial")

plotdata = @_ statdata |>
    @transform(__, compname = map(x -> compnames[x], :comparison)) |>
    groupby(__, [:target_time_label, :compname, :comparison]) |>
    @combine(__,
        mean = mean(:mean),
        lower = lowerboot(:mean),
        upper = upperboot(:mean),
    ) |>
    @transform(__, comptime = string.(:compname, :target_time_label))

nullmean = statdata.logitnullmean |> mean |> logistic
ytitle = "Condition Classification"
barwidth = 16

time_comp = OrderedDict(
    "mix1_2_early" => GermanTrack.colorat([1,7]),
    "mix1_2_late" => GermanTrack.colorat([3,9]),
    "mix1_3_early" => GermanTrack.colorat([1,12]),
    "mix1_3_late" => GermanTrack.colorat([3,14]),
    "mix2_3_early" => GermanTrack.colorat([7,12]),
    "mix2_3_late" => GermanTrack.colorat([9,14]),
)

plhit = @_ plotdata |>
    @vlplot(
        # facet = { column = { field = :target_time_label, type = :nominal} },
        width = 230, height = 130,
        config = {
            bar = {discreteBandSize = barwidth},
            axis = {labelFont = "Helvetica", titleFont = "Helvetica"},
            legend = {disable = true, labelFont = "Helvetica", titleFont = "Helvetica"},
            header = {labelFont = "Helvetica", titleFont = "Helvetica"},
            mark = {font = "Helvetica"},
            text = {font = "Helvetica"},
            title = {font = "Helvetica", subtitleFont = "Helvetica"}
        }
    ) + (
    @vlplot(x = {:compname, axis = {
            labelAngle = 0,
            title = "",
            labelExpr = "split(datum.label, '\\n')"}},
        color = {
            :comptime, title = nothing,
            scale = {range = urlcol.(keys(time_comp))}}) +
    @vlplot({:bar, xOffset = -barwidth/2 - 1, clip = true},
        transform = [{filter = "datum.target_time_label == 'early'"}],
        y = {:mean,
            scale = {domain = [0.5, 1]},
            title = ytitle}) +
    @vlplot({:rule, xOffset = -barwidth/2 - 1},
        transform = [{filter = "datum.target_time_label == 'early'"}],
        color = {value = "black"},
        y2 = :upper,
        y = {:lower,
            scale = {domain = [0.5, 1]},
            title = ytitle}) +
    @vlplot({:bar, xOffset = barwidth/2 + 1, clip = true},
        transform = [{filter = "datum.target_time_label == 'late'"}],
        y = {:mean,
            scale = {domain = [0.5, 1]},
            title = ytitle}) +
    @vlplot({:rule, xOffset = barwidth/2 + 1},
        transform = [{filter = "datum.target_time_label == 'late'"}],
        color = {value = "black"},
        y2 = :upper,
        y = {:lower,
            scale = {domain = [0.5, 1]},
            title = ytitle})
    ) +
    @vlplot({:text, angle = -90, fontSize = 9, align = "left", baseline = "bottom", dx = 0, dy = -barwidth-2},
        transform = [{filter = "datum.target_time_label == 'early' && datum.comparison == 'global-v-object'"}],
        # x = {datum = "spatial"}, y = {datum = 0.},
        x = :compname,
        y = {datum = 0.5},
        text = {value = "Early"},
        ) +
    @vlplot({:text, angle = -90, fontSize = 9, align = "left", baseline = "top", dx = 0, dy = barwidth+2},
        transform = [{filter = "datum.target_time_label == 'late' && datum.comparison == 'global-v-object'"}],
        # x = {datum = "spatial"}, y = {datum = },
        x = :compname,
        y = {datum = 0.5},
        text = {value = "Late"},
    );
plotfile = joinpath(dir, "supplement", "condition_earlylate.svg")
plhit |> save(plotfile)
addpatterns(plotfile, time_comp, size = 10)

# Categorize the target for each condition
# =================================================================

# Find best λs
# -----------------------------------------------------------------

file = joinpath(processed_datadir("analyses"), "target_lambdas.json")
GermanTrack.@cache_results file fold_map hyperparams begin
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    offsets = range(0, 1.0, length = 10)
    classdf = @_ events |>
        filter(ishit(_) == "hit", __) |>
        groupby(__, [:sid, :condition]) |>
        filteringmap(__, desc = "Computing features...",
        :window => [windowtarget(start = start, len = len)
            for len in 2.0 .^ range(-4, 1, length = 10),
                start in offsets],
            compute_powerbin_features(_1, subjects, _2)) |>
        deletecols!(__, :window) |>
        transform!(__, :windowtype => (x -> "target") => :windowtype)

    classbasedf = @_ events |>
        filter(ishit(_) == "hit", __) |>
        groupby(__, [:sid, :condition]) |>
        filteringmap(__, desc = "Computing features...",
        :window => [windowtarget(start = -len, len = len)
            for len in 2.0 .^ range(-4, 1, length = 10)],
            compute_powerbin_features(_1, subjects, _2)) |>
        transform!(__, :windowtype => (x -> "baseline") => :windowtype) |>
        deletecols!(__, :window) |>
        append!!(classdf, __)

    lambdas = 10.0 .^ range(-2, 0, length = 100)
    resultdf = @_ classbasedf |>
        addfold!(__, 10, :sid, rng = stableRNG(2019_11_18, :target_lambda_folds)) |>
        groupby(__, [:winlen, :condition]) |>
        filteringmap(__, desc = "Evaluating lambdas...", folder = foldxt,
            :cross_fold => 1:10,
            :winstart => offsets,
            function(sdf, fold, offset)
                target = @where(sdf, (:winstart .== offset) .& (:windowtype .== "target"))
                baseline = @where(sdf, :windowtype .== "baseline")
                rng = stableRNG(2019_11_18, :validate_lambda, fold, sdf.condition)
                test, model = traintest(vcat(target, baseline), fold, y = :windowtype,
                    λ = lambdas, selector = m -> AllSeg(), validate_rng = rng)
                test[:, Not(r"channel")]
            end)

    fold_map = @_ resultdf |>
        groupby(__, :sid) |> combine(__, :fold => first => :fold) |>
        Dict(row.sid => row.fold for row in eachrow(__))

    win_means = @_ resultdf |>
        groupby(__, [:winlen, :winstart, :λ, :fold]) |>
        @combine(__, mean = mean(:val_accuracy), mean_sem = mean(:val_se))

    hyperparams = @_ win_means |>
        groupby(__, :fold) |>
        @combine(__,
            best = maximum(:mean),
            λ = :λ[argmax(:mean)],
            winlen = :winlen[argmax(:mean)],
            winstart = :winstart[argmax(:mean)]
        ) |>
        Dict(row.fold => (;row[Not(:fold)]...) for row in eachrow(__))

    @info "Saving plots to $(joinpath(dir, "supplement"))"
end

# Target classification timeline
# -----------------------------------------------------------------

file = joinpath(processed_datadir("analyses"), "target_results.json")
GermanTrack.@cache_results file resultdf begin
    lens = @_ getindex.(values(hyperparams), :winlen) |> unique |>
        GermanTrack.spread.(__, 0.5.*__, n_winlens) |> reduce(vcat, __) |> unique

    offsets = range(-1.0, 4.0, length = 32)
    classdf = @_ events |>
        filter(ishit(_) == "hit", __) |>
        groupby(__, [:sid, :condition]) |>
        filteringmap(__, desc = "Computing features...",
        :window => [windowtarget(start = start, len = len)
            for len in lens,
                start in offsets],
            compute_powerbin_features(_1, subjects, _2)) |>
        transform!(__, :windowtype => (x -> "target") => :windowtype) |>
        deletecols!(__, :window)

    classbasedf = @_ events |>
        filter(ishit(_) == "hit", __) |>
        groupby(__, [:sid, :condition]) |>
        filteringmap(__, desc = "Computing features...",
        :window => [windowtarget(start = -len, len = len)
            for len in lens],
            compute_powerbin_features(_1, subjects, _2)) |>
        transform!(__, :windowtype => (x -> "baseline") => :windowtype) |>
        deletecols!(__, :window) |>
        append!!(classdf, __)

    zero_tolerance = 1e-3

    function train_sets(target_offset)
        function(train)
            sids = unique(train.sid)
            pretarget = train.winstart[train.winstart .< 0.0]
            rand_1 = @_ sample(pretarget, length(sids), replace = false)
            rand_2 = @_ sample(pretarget, length(sids), replace = false)

            real_train = @_ filter(_1.winstart == target_offset || _1.windowtype == "baseline", train)
            real_train.istarget = real_train.windowtype == "target"

            start_map = Dict(sids .=> rand_1)
            baseline_map = Dict(sids .=> rand_2)
            baseline_train = @_ filter(
                _1.winstart ∈ (start_map[_1.sid], baseline_map[_1.sid]) &&
                _1.windowtype == "target", train)
            baseline_train.istarget = @_ map(baseline_map[_1.sid] == _1.winstart,
                eachrow(baseline_train))

            @infiltrate

            real_train, baseline_train
        end
    end

    resultdf = @_ classdf |>
        transform!(__, :sid => ByRow(x -> fold_map[x]) => :fold) |>
        groupby(__, [:condition]) |>
        filteringmap(__, desc = "Evaluating lambdas...", folder = foldl,
            :cross_fold => 1:10,
            :modeltype => ["full", "null"],
            function(sdf, fold, modeltype)
                target_offset = hyperparams[fold][:winstart]
                target_offset = offsets[argmin(abs.(target_offset .- offsets))]
                sdf.istarget = sdf.winstart .== target_offset

                len = hyperparams[fold][:winlen]
                lens = GermanTrack.spread(len, 0.5 * len, n_winlens)
                sdf_ = filter(x -> x.winlen ∈ lens, sdf)

                # train on random windows before
                combine(groupby(sdf_, :winlen)) do sdf_len
                    train, baseline = filter(x -> x.fold != fold, sdf_len) |>
                        train_sets(target_offset)
                    test = filter(x -> x.fold == fold, sdf_len)
                    # we pretend all test offsets are targets, so we get a measure of how often
                    # each offset is calculated as a target
                    test.istarget = true

                    test, model = traintest(sdf_len, fold,
                        y = :istarget,
                        selector = modeltype == "full" ? hyperparams[fold][:λ] :
                                m -> NullSelect(),
                        train_test = (train,test)
                    )
                    test.train_type = "target"

                    btest, bmodel = traintest(sdf_len, fold,
                        y = :istarget,
                        selector = modeltype == "full" ? hyperparams[fold][:λ] :
                                m -> NullSelect(),
                        train_test = (baseline,copy(test))
                    )
                    btest.train_type = "baseline"

                    if maximum(abs.(Array(view(test,:, r"channel")) .- 0.0)) < zero_tolerance
                        Empty(DataFrame)
                    else
                        vcat(
                            view(test,:, Not(r"channel")),
                            view(btest, :, Not(r"channel"))
                        )
                    end
                end
            end)
end

# Plotting
# -----------------------------------------------------------------

classmeans = @_ resultdf |>
    @where(__, :modeltype .== "full") |>
    groupby(__, [:winstart, :winlen, :sid, :fold, :condition, :train_type]) |>
    combine(__, :correct => mean => :correct,
                :correct => length => :count) |>
    groupby(__, [:winstart, :sid, :fold, :condition, :train_type]) |>
    combine(__, :correct => mean => :correct,
                :correct => mean => :count) |>
    unstack(__, [:winstart, :sid, :fold, :condition], :train_type, :correct)

logitbasemean = mean(logit.(shrink.(classmeans.baseline)))
basemean = logistic.(logitbasemean)

plotdata = @_ classmeans |>
    @transform(__,
        corrected_mean =
            logistic.(logit.(shrink.(:target)) .-
                logit.(shrink.(:baseline)) .+
                logitbasemean),
        condition_label = uppercasefirst.(:condition)
    )

ytitle = "Target Classification"
target_len_y = 0.8
label_x = plotdata.winstart |> maximum
pl = @_ plotdata |>
    groupby(__, [:condition, :condition_label, :winstart]) |>
    @combine(__,
        corrected_mean = mean(:corrected_mean),
        lower = lowerboot(:corrected_mean, alpha = 0.318),
        upper = upperboot(:corrected_mean, alpha = 0.318),
    ) |>
    @vlplot(
        width = 130, height = 140,
        config = {
            axis = {labelFont = "Helvetica", titleFont = "Helvetica"},
            legend = {disable = true, labelFont = "Helvetica", titleFont = "Helvetica"},
            header = {labelFont = "Helvetica", titleFont = "Helvetica"},
            mark = {font = "Helvetica"},
            text = {font = "Helvetica"},
            title = {font = "Helvetica", subtitleFont = "Helvetica"}
        },
    ) +
    (@vlplot(
        color = {field = :condition, type = :nominal,
            scale = {range = "#".*hex.(colors)}},
    ) +
    # data lines
    @vlplot({:line, strokeCap = :round, clip = true},
        strokeDash = {:condition, type = :nominal, scale = {range = [[1, 0], [6, 4], [2, 4]]}},
        x = {:winstart, type = :quantitative, title = "Time relative to target onset (s)"},
        y = {:corrected_mean, aggregate = :mean, type = :quantitative, title = ytitle,
            scale = {domain = [0,1.0]}}) +
    # data errorbands
    @vlplot({:errorband, clip = true},
        x = {:winstart, type = :quantitative},
        y = {:lower, type = :quantitative, title = ytitle},
        y2 = :upper
    ) +
    # condition labels
    @vlplot({:text, align = :left, dx = 5},
        transform = [{filter =
            "(datum.winstart > 3.6 && datum.winstart <= 3.95 && "*
                "datum.condition != 'object') ||"*
            "(datum.winstart > 3.2 && datum.winstart <= 3.4 && "*
                "datum.condition == 'object')"}],
        x = {datum = label_x},
        y = {:corrected_mean, aggregate = :mean, type = :quantitative},
        text = :condition_label
    ) +
    # "Null Model" text annotation
    (
        @vlplot(data = {values = [{}]}) +
        # white rectangle to give text a background
        @vlplot(mark = {:text, size = 11, baseline = "top", dy = 2, dx = 0,
            align = "center"},
            x = {datum = mean(offsets)}, y = {datum = nullmean},
            text = {value = ["Baseline", "Accuracy"]},
            color = {value = "black"}
        )
    ) +
    # Dotted line
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:rule, strokeDash = [4 4], size = 2},
            y = {datum = nullmean},
            color = {value = "black"})
    ) +
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
        @vlplot(mark = {:text, size = 11, baseline = "bottom", align = :left, yOffset = -3},
            x = {datum = 0}, y = {datum = target_len_y},
            text = {value = "Target Length"},
            color = {value = "black"}
        )
    ));
pl |> save(joinpath(dir, "fig2d.svg"))

# EEG Cross-validated Features
# =================================================================

GermanTrack.@cache_results file coefdf begin
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    start_len = unique((x.winstart, x.winlen) for x in values(hyperparams))

    classdf = @_ events |>
        transform!(__, AsTable(:) => ByRow(ishit) => :hittype) |>
        @where(__, :hittype .== "hit") |>
        transform!(__, :sid => ByRow(x -> fold_map[x]) => :fold) |>
        groupby(__, [:sid, :condition]) |>
        filteringmap(__, desc = "Computing features...",
            :window => [windowtarget(start = start, len = len)
                for (start, len) in start_len],
            compute_powerbin_features(_1, subjects, _2)) |>
        deletecols!(__, :window)

    coefdf = @_ classdf |>
        addfold!(__, 10, :sid, rng = stableRNG(2019_11_18, :condition_lambda_folds)) |>
        filteringmap(__, desc = "Evaluating lambdas...", folder = foldxt,
            :cross_fold => 1:10,
            :comparison => (
                "global-v-object"  => x -> x.condition ∈ ["global", "object"],
                "global-v-spatial" => x -> x.condition ∈ ["global", "spatial"],
                "object-v-spatial" => x -> x.condition ∈ ["object", "spatial"],
            ),
            function(sdf, fold, comparison)
                conds = split(comparison, "-v-")

                test, model = traintest(sdf, fold, y = :condition)

                coefs = coef(model, MinAICc())'
                DataFrame(coefs, vcat("I", names(view(sdf, r"channel"))))
            end)

    longdf = @_ coefdf |>
        stack(__, All(r"channel"), [:cross_fold, :comparison]) |>
        @transform(__,
            channel = parse.(Int, getindex.(match.(r"channel_([0-9]+)", :variable), 1)),
            freqbin = getindex.(match.(r"channel_[0-9]+_(.*)$", :variable), 1)
        )

end

pl = @_ longdf |>
    groupby(__, [:freqbin, :comparison, :cross_fold]) |>
    @combine(__, value = sum(!iszero, :value)) |>
    groupby(__, [:freqbin, :comparison]) |>
    @combine(__,
        value = median(:value),
        lower2 = quantile(:value, 0.05),
        upper2 = quantile(:value, 0.95),
        lower1 = quantile(:value, 0.25),
        upper1 = quantile(:value, 0.75)
    ) |>
    @vlplot(
        facet = {column = {field = :comparison, type = :nominal}}
    ) + (
        @vlplot() +
        (
            @vlplot(x = {:freqbin,
                sort = string.(keys(GermanTrack.default_freqbins)),
                type = :ordinal,
            }) +
            @vlplot(:rule, y = :lower2, y2 = :upper2) +
            @vlplot({:rule, size = 4}, y = :lower1, y2 = :upper1) +
            @vlplot({:point, size = 50, filled = true}, y = :value, color = {value = "black"})
        )
    );
pl |> save(joinpath(dir, "fig2c.svg"))

# Combine Figures (Full Figure 2)
# =================================================================

GermanTrack.@usepython

svg = pyimport("svgutils").compose

background_file = tempname()

background = pyimport("svgutils").transform.fromstring("""
    <svg>
        <rect width="100%" height="100%" fill="white"/>
    </svg>
""").save(background_file)

fig = svg.Figure("174mm", "67mm",
    svg.SVG(background_file),
    svg.Panel(
        svg.SVG(joinpath(dir, "fig2a.svg")).move(0,15),
        svg.Text("A", 2, 10, size = 12, weight="bold", font = "Helvetica"),
        svg.SVG(joinpath(plotsdir("icons"), "behavior.svg")).
            scale(0.1).move(210,15)
    ).move(0, 0),
    svg.Panel(
        svg.SVG(joinpath(dir, "fig2b.svg")).move(0,15),
        svg.Text("B", 2, 10, size = 12, weight = "bold", font = "Helvetica"),
        svg.SVG(joinpath(plotsdir("icons"), "eeg.svg")).
            scale(0.1).move(220,15)
    ).move(240, 0)
).scale(1.333).save(joinpath(plotsdir("figures"), "fig2.svg"))




