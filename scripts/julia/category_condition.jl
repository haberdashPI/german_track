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
        transform!(__, AsTable(:) => ByRow(x -> ishit(x, region = "target")) => :hittype) |>
        @where(__, :hittype .∈ Ref(["hit", "miss"])) |>
        groupby(__, [:sid, :condition, :hittype]) |>
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
            :compare_hit => [true, false],
            :comparison => (
                "global-v-object"  => x -> x.condition ∈ ["global", "object"],
                "global-v-spatial" => x -> x.condition ∈ ["global", "spatial"],
                "object-v-spatial" => x -> x.condition ∈ ["object", "spatial"],
            ),
            function(sdf, fold, usehit, comparison)
                sdf.complabel = (sdf.condition .== first(sdf.condition)) .&
                    (.!usehit .| (sdf.hittype .== "hit"))
                test, model = traintest(sdf, fold, y = :complabel, weight = :weight)
                test.nzero = sum(!iszero, coef(model, MinAICc()))
                test
            end)

    fold_map = @_ resultdf |>
        groupby(__, :sid) |> combine(__, :fold => first => :fold) |>
        Dict(row.sid => row.fold for row in eachrow(__))

    win_means = @_ resultdf |>
        groupby(__, [:comparison, :winlen, :winstart, :sid, :fold, :compare_hit]) |>
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
        @where(__, :hittype .∈ Ref(["hit", "miss"])) |>
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
        filteringmap(__, desc = "Classifying conditions...", folder = foldxt,
            :cross_fold => 1:10,
            :comparison => (
                "global-v-object"  => x -> x.condition ∈ ["global", "object"],
                "global-v-spatial" => x -> x.condition ∈ ["global", "spatial"],
                "object-v-spatial" => x -> x.condition ∈ ["object", "spatial"],
            ),
            :compare_hit => [true, false],
            :modeltype => modeltypes,
            function (sdf, fold, comparison, usehit, modeltype)
                selector = modeltype == "null" ? m -> NullSelect() : m -> MinAICc()
                sdf = get(modelshufflers, modeltype, identity)(sdf)

                win = hyperparams[fold]
                sdf = filter(x -> x.winstart == win.winstart && x.winlen == win.winlen, sdf)

                sdf.complabel = (sdf.condition .== first(sdf.condition)) .&
                    (.!usehit .| (sdf.hittype .== "hit"))
                test, model = traintest(sdf, fold, y = :complabel,
                    selector = selector)

                if modeltype == "null"
                    test.nzero = sum(!iszero, coef(model, NullSelect()))
                else
                    test.nzero = sum(!iszero, coef(model, MinAICc()))
                end

                test
            end)
end

# Main EEG results (Figure 2B)
# =================================================================

predictmeans = @_ predictbasedf |>
    filter(_.modeltype ∈ ["null", "full"], __) |>
    groupby(__, [:sid, :comparison, :compare_hit, :modeltype]) |>
    @combine(__,
        correct = mean(:correct),
        n = length(:correct),
    )

file = joinpath(processed_datadir("analyses"), "eeg_condition.csv")
CSV.write(file, predictmeans)
run(`Rscript $(joinpath(scriptsdir("R"), "condition_eeg.R"))`)

compnames = OrderedDict(
    "global_v_object"  => "Global vs.\n Object",
    "global_v_spatial" => "Global vs.\n Spatial",
    "object_v_spatial" => "Object vs.\n Spatial")

statfile = joinpath(processed_datadir("analyses"), "eeg_condition_coefs.csv")
plotdata = @_ CSV.read(statfile, DataFrame) |>
    rename(__, :propr_med => :mean, :propr_05 => :lower, :propr_95 => :upper) |>
    @transform(__,
        label = :comparison,
        comparison = string.(getindex.(match.(r"(.*)_(all|hit)", :comparison), 1)),
        case = string.(getindex.(match.(r"(.*)_(all|hit)", :comparison), 2)),
    ) |>
    transform!(__, :comparison => ByRow(x -> compnames[x]) => :compname)
nullmean = @_ predictmeans |> filter(_.modeltype == "null", __) |> __.correct |> mean

plcolors = OrderedDict(
    "mix1_2_early" => GermanTrack.colorat([1,7]),
    "mix1_2_late" => GermanTrack.colorat([3,9]),
    "mix1_3_early" => GermanTrack.colorat([1,12]),
    "mix1_3_late" => GermanTrack.colorat([3,14]),
    "mix2_3_early" => GermanTrack.colorat([7,12]),
    "mix2_3_late" => GermanTrack.colorat([9,14]),
)

ytitle = "Condition Classification"
barwidth = 14
plhit = @_ plotdata |>
    @vlplot(
        # facet = { column = { field = :case, type = :nominal} },
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
            :label, title = nothing,
            scale = {range = urlcol.(keys(plcolors))}}) +
    @vlplot({:bar, xOffset = -barwidth/2, clip = true},
        transform = [{filter = "datum.case == 'all'"}],
        y = {:mean,
            scale = {domain = [0.5, 1]},
            title = ytitle}) +
    @vlplot({:rule, xOffset = -barwidth/2},
        transform = [{filter = "datum.case == 'all'"}],
        color = {value = "black"},
        y2 = :upper,
        y = {:lower,
            scale = {domain = [0.5, 1]},
            title = ytitle}) +
    @vlplot({:bar, xOffset = barwidth/2, clip = true},
        transform = [{filter = "datum.case == 'hit'"}],
        y = {:mean,
            scale = {domain = [0.5, 1]},
            title = ytitle}) +
    @vlplot({:rule, xOffset = barwidth/2},
        transform = [{filter = "datum.case == 'hit'"}],
        color = {value = "black"},
        y2 = :upper,
        y = {:lower,
            scale = {domain = [0.5, 1]},
            title = ytitle})
    ) +
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:rule, strokeDash = [4 4], size = 1},
            y = {datum = nullmean},
            color = {value = "black"}) +
        @vlplot({:text, align = "left", dx = 15, dy = 0, baseline = "line-bottom", fontSize = 9},
            y = {datum = nullmean},
            x = {datum = "Object vs.\n Spatial"},
            text = {value = ["Null Model", "Accuracy"]}
        )
    ) +
    @vlplot({:text, angle = -90, fontSize = 9, align = "left", baseline = "bottom", dx = 0, dy = -barwidth-2},
        transform = [{filter = "datum.case == 'all' && datum.comparison == 'global_v_object'"}],
        x = :compname,
        y = {datum = 0.5},
        text = {value = "Hits & Misses"}) +
    @vlplot({:text, angle = -90, fontSize = 9, align = "left", baseline = "top", dx = 0, dy = barwidth+2},
        transform = [{filter = "datum.case == 'hit' && datum.comparison == 'global_v_object'"}],
        x = :compname,
        y = {datum = 0.5},
        text = {value = "Hits"});
plhit |> save(joinpath(dir, "fig2b.vegalite"))
plotfile = joinpath(dir, "fig2b.svg")
plhit |> save(plotfile)
addpatterns(plotfile, plcolors, size = 10)

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

# Condition timeline
# =================================================================

subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

lens = @_ getindex.(values(hyperparams), :winlen) |> unique |>
GermanTrack.spread.(__, 0.5, n_winlens) |> reduce(vcat, __) |> unique

classdf = @_ events |>
    transform!(__, AsTable(:) => ByRow(x -> ishit(x, region = "target")) => :hittype) |>
    @where(__, :hittype .∈ Ref(["hit", "miss"])) |>
    groupby(__, [:sid, :condition, :hittype]) |>
    filteringmap(__, desc = "Computing features...",
        :window => [windowtarget(len = len, start = start)
            for len in lens, start in range(-4, 4, length = 32)],
        compute_powerbin_features(_1, subjects, _2)) |>
    transform!(__, :sid => ByRow(x -> fold_map[x]) => :fold) |>
    deletecols!(__, :window)

resultdf = @_ classdf |>
    groupby(__, [:winstart]) |>
    filteringmap(__, desc = "Evaluating lambdas...", folder = foldxt,
        :cross_fold => 1:10,
        :compare_hit => [true, false],
        :modeltype => ["full", "null"],
        :comparison => (
            "global-v-object"  => x -> x.condition ∈ ["global", "object"],
            "global-v-spatial" => x -> x.condition ∈ ["global", "spatial"],
            "object-v-spatial" => x -> x.condition ∈ ["object", "spatial"],
        ),
        function(sdf, fold, usehit, modeltype, comparison)
            selector = modeltype == "null" ? m -> NullSelect() : m -> MinAICc()
            sdf.complabel = (sdf.condition .== first(sdf.condition)) .&
                (.!usehit .| (sdf.hittype .== "hit"))
            lens = hyperparams[fold][:winlen] |> GermanTrack.spread(0.5, n_winlens)

            sdf = filter(x -> x.winlen ∈ lens, sdf)
            combine(groupby(sdf, :winlen)) do sdf_len
                test, model = traintest(sdf_len, fold, y = :complabel,
                    selector = selector)
                test.nzero = sum(!iszero, coef(model, MinAICc()))
                test[:, Not(r"channel")]
            end
        end)

# plotting
# -----------------------------------------------------------------

classmeans = @_ resultdf |>
    # @where(__, :modeltype .== "full") |>
    groupby(__, [:winstart, :winlen, :sid, :fold, :comparison, :compare_hit, :modeltype]) |>
    combine(__, :correct => mean => :correct,
                :correct => length => :count) |>
    groupby(__, [:winstart, :sid, :fold, :comparison, :compare_hit, :modeltype]) |>
    combine(__, :correct => mean => :correct,
                :correct => mean => :count) |>
    unstack(__, [:winstart, :sid, :fold, :comparison, :compare_hit], :modeltype, :correct)

plotdata = @_ classmeans |>
    groupby(__, [:compare_hit]) |>
    @transform(__, logitnullmean = mean(logit.(shrink.(:null)))) |>
    groupby(__, [:compare_hit]) |>
    @transform(__,
        nullmean = logistic.(:logitnullmean),
        corrected_mean =
            logistic.(logit.(shrink.(:full)) .-
                logit.(shrink.(:null)) .+
                :logitnullmean))

ytitle = "Target Classification"
target_len_y = 0.8
label_x = plotdata.winstart |> maximum
pl = @_ plotdata |>
    groupby(__, [:comparison, :winstart, :compare_hit]) |>
    @combine(__,
        corrected_mean = mean(:corrected_mean),
        nullmean = mean(:nullmean),
        lower = lowerboot(:corrected_mean, alpha = 0.318),
        upper = upperboot(:corrected_mean, alpha = 0.318),
    ) |>
    @vlplot(
        width = 130, height = 140,
        facet = {column = {field = :compare_hit}},
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
        color = {field = :comparison, type = :nominal,
            scale = {range = "#".*hex.(colors)}},
    ) +
    # data lines
    @vlplot({:line, strokeCap = :round, clip = true},
        strokeDash = {:comparison, type = :nominal, scale = {range = [[1, 0], [6, 4], [2, 4]]}},
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
        transform = [{filter = "(datum.winstart > 3.6 && datum.winstart <= 3.95)"}],
        x = {datum = label_x},
        y = {:corrected_mean, aggregate = :mean, type = :quantitative},
        text = :comparison
    ) +
    # "Null Model" text annotation
    # white rectangle to give text a background
    @vlplot(mark = {:text, size = 11, baseline = "top", dy = 2, dx = 0,
        align = "center"},
        x = "mean(winstart)", y = "mean(nullmean)",
        text = {value = ["Baseline", "Accuracy"]},
        color = {value = "black"}
    ) +
    # Dotted line
    @vlplot(mark = {:rule, strokeDash = [4 4], size = 2},
        y = "mean(nullmean)",
        color = {value = "black"}) +
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
pl |> save(joinpath(dir, "fig2d_A.svg"))


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




