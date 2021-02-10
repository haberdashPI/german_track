# Setup
# =================================================================

using DrWatson; @quickactivate("german_track")

using GermanTrack, DataFrames, Statistics, Dates, Underscores, Random, Printf,
    ProgressMeter, VegaLite, FileIO, StatsBase, BangBang, Transducers, Infiltrator, Peaks,
    StatsFuns, Distributions, DSP, DataStructures, Colors, Bootstrap, CSV, EEGCoding
wmean = GermanTrack.wmean
n_winlens = 6

dir = mkpath(joinpath(plotsdir(), "figure2_parts", "supplement"))

using GermanTrack: colors, neutral, patterns

# Behavioral Analyses
# =================================================================

# Hit rates for EEG experiment
# -----------------------------------------------------------------

subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

events = @_ events |>
    transform(__, AsTable(:) => ByRow(findresponse) => :hittype)

rates = @_ events |>
    transform(__, AsTable(:) =>
        ByRow(findresponse) => :hittype) |>
    groupby(__, [:condition, :sid]) |>
    @combine(__,
        hit = sum(:hittype .== "hit") / sum(:hittype .∈ Ref(Set(["hit", "miss"]))),
        falsep = sum(:hittype .== "falsep") /
                 sum(:hittype .∈ Ref(Set(["falsep", "reject"])))
    )

bad_sids = @_ rates |>
    @where(__, :condition .== "global") |>
    @where(__, (:hit .<= :falsep) .| (:falsep .>= 1)) |> __.sid |> Ref |> Set

means = @_ rates |>
    @where(__, :sid .∉ bad_sids) |>
    stack(__, [:hit, :falsep], [:condition, :sid],
        variable_name = :type, value_name = :proportion) |>
    groupby(__, [:condition, :type]) |>
    combine(__, :proportion => mean => :prop,
                :proportion => lowerboot => :lower,
                :proportion => upperboot => :upper)

# TODO: apply beta-reg stats

barwidth = 20
means |> @vlplot(
        width = 242, autosize = "fit",
        # width = {step = 50},
        config = {
            legend = {disable = true},
            bar = {discreteBandSize = barwidth}
        }) +
    @vlplot({:bar, xOffset = -(barwidth/2)},
        transform = [{filter = "datum.type == 'hit'"}],
        x = {:condition, axis = {title = "", labelAngle = 0,
            labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}, },
        y = {:prop, type = :quantitative, aggregate = :mean,
                scale = {domain = [0, 1]}, title = "Response Rate"},
        color = {:condition, scale = {range = "#".*hex.(colors)}}) +
    @vlplot({:bar, xOffset = (barwidth/2)},
        transform = [{filter = "datum.type == 'falsep'"}],
        x = {:condition, axis = {title = ""}},
        y = {:prop, type = :quantitative, aggregate = :mean},
        color = {value = "#"*hex(neutral)}) +
    @vlplot({:rule, xOffset = -(barwidth/2)},
        transform = [{filter = "datum.type == 'hit'"}],
        color = {value = "black"},
        x = {:condition, axis = {title = ""}},
        y = {:lower, title = ""}, y2 = :upper
    ) +
    @vlplot({:rule, xOffset = (barwidth/2)},
        transform = [{filter = "datum.type == 'falsep'"}],
        color = {value = "black"},
        x = {:condition, axis = {title = ""}},
        y = {:lower, title = ""}, y2 = :upper
    ) +
    @vlplot({:text, angle = -90, fontSize = 9, align = "right", baseline = "bottom", dx = 0, dy = -barwidth-2},
        transform = [{filter = "datum.condition == 'global' && datum.type == 'hit'"}],
        # x = {datum = "spatial"}, y = {datum = 0.},
        x = {:condition, axis = {title = ""}},
        y = {:prop, aggregate = :mean, type = :quantitative},
        text = {value = "Hits"},
    ) +
    @vlplot({:text, angle = -90, fontSize = 9, align = "left", baseline = "top", dx = 0, dy = barwidth+2},
        transform = [{filter = "datum.condition == 'global' && datum.type == 'falsep'"}],
        # x = {datum = "spatial"}, y = {datum = },
        x = {:condition, axis = {title = ""}},
        y = {datum = 0},
        text = {value = "False Positives"},
    ) |>
    save(joinpath(dir, "supplement", "eeg_behavior_hitrate.svg"))

# Hit rate by experiment
# -----------------------------------------------------------------

means_byexp = @_ CSV.read(joinpath(processed_datadir("analyses"),
    "behavioral_condition.csv")) |>
    transform!(__, :block_type => ByRow(x -> ascondition[x]) => :condition) |>
    rename(__,:sbj_id => :sid) |>
    select(__, :condition, :sid, :hr, :fr, :exp_id) |>
    stack(__, [:hr, :fr], [:condition, :sid, :exp_id], variable_name = :type, value_name = :prop) |>
    groupby(__, [:condition, :type, :exp_id]) |>
    combine(__,
        :prop => mean => :prop,
        :prop => lowerboot => :lower,
        :prop => upperboot => :upper
    )

barwidth = 20
means_byexp |> @vlplot(
    facet = {field = :exp_id, type = :nominal, title = "",
        header = {labelExpr = "'Experiment '+datum.label"}},
    config = {
        legend = {disable = true},
        bar = {discreteBandSize = barwidth},
        facet = {columns = 3}
    }) + (
        @vlplot(
            width = {step = 60},
        ) +
        @vlplot({:bar, xOffset = -(barwidth/2)},
            transform = [{filter = "datum.type == 'hr'"}],
            x = {:condition, axis = {title = "", labelAngle = 0,
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}, },
            y = {:prop, type = :quantitative, aggregate = :mean,
                    scale = {domain = [0, 1]}, title = "Response Rate"},
            color = {:condition, scale = {range = "#".*hex.(colors)}}) +
        @vlplot({:bar, xOffset = (barwidth/2)},
            transform = [{filter = "datum.type == 'fr'"}],
            x = {:condition, axis = {title = ""}},
            y = {:prop, type = :quantitative, aggregate = :mean},
            color = {value = "#"*hex(neutral)}) +
        @vlplot({:rule, xOffset = -(barwidth/2)},
            transform = [{filter = "datum.type == 'hr'"}],
            color = {value = "black"},
            x = {:condition, axis = {title = ""}},
            y = {:lower, title = ""}, y2 = :upper
        ) +
        @vlplot({:rule, xOffset = (barwidth/2)},
            transform = [{filter = "datum.type == 'fr'"}],
            color = {value = "black"},
            x = {:condition, axis = {title = ""}},
            y = {:lower, title = ""}, y2 = :upper
        ) +
        @vlplot({:text, angle = -90, fontSize = 9, align = "right", baseline = "bottom", dx = 0, dy = -barwidth-2},
            transform = [{filter = "datum.condition == 'global' && datum.type == 'hr'"}],
            # x = {datum = "spatial"}, y = {datum = 0.},
            x = {:condition, axis = {title = ""}},
            y = {:prop, aggregate = :mean, type = :quantitative},
            text = {value = "Hits"},
        ) +
        @vlplot({:text, angle = -90, fontSize = 9, align = "left", baseline = "top", dx = 0, dy = barwidth+2},
            transform = [{filter = "datum.condition == 'global' && datum.type == 'fr'"}],
            # x = {datum = "spatial"}, y = {datum = },
            x = {:condition, axis = {title = ""}},
            y = {datum = 0},
            text = {value = "False Positives"},
        )
    ) |> save(joinpath(dir, "raw_sum_behavior_byexp.svg"))


# Hit rate computed from raw data
# -----------------------------------------------------------------

info = GermanTrack.load_behavioral_stimulus_metadata()

events = @_ readdir(processed_datadir("behavioral"), join=true) |>
    filter(occursin(r"csv$", _), __) |>
    mapreduce(GermanTrack.events(_, info), append!!, __)

indmeans = @_ events |>
    transform!(__, AsTable(:) =>
        ByRow(x -> findresponse(x, region = "target", mark_false_targets = true)) => :hittype) |>
    groupby(__, [:condition, :sid]) |>
    combine(__, :hittype => (x -> mean(==("hit"), x)) => :hits,
                :hittype => (x -> mean(y -> occursin("falsep", y), x)) => :falseps,
                :hittype => (x -> mean(==("falsep"), x)) => :notargets,
                :hittype => (x -> mean(==("falsep-target"), x)) => :falsetargets)

bad_sids = @_ indmeans |>
    filter(_.condition != "global", __) |>
    groupby(__, :sid) |>
    combine(__, [:hits, :falseps] => ((x, y) -> minimum(x - y)) => :diffs) |>
    filter(_.diffs < 0, __) |>
    __.sid

CSV.write(joinpath(processed_datadir("behavioral", "outliers"), "sids.csv"), DataFrame(sid = bad_sids))

means = @_ indmeans |>
    filter(_.sid ∉ bad_sids, __) |>
    stack(__, [:hits, :falseps, :notargets, :falsetargets], [:condition, :sid],
        variable_name = :type, value_name = :proportion) |>
    groupby(__, [:condition, :type]) |>
    combine(__, :proportion => mean => :prop,
                :proportion => (x -> lowerboot(x, alpha = 0.05)) => :lower,
                :proportion => (x -> upperboot(x, alpha = 0.05)) => :upper)

means |> @vlplot(
    width = {step = 50},
    config = {
        legend = {disable = true},
        bar = {discreteBandSize = 16}
    }) +
@vlplot({:bar, xOffset = -8},
    transform = [{filter = "datum.type == 'hits'"}],
    x = {:condition, axis = {title = "", labelAngle = 0,
        labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}, },
    y = {:prop, type = :quantitative, aggregate = :mean,
         scale = {domain = [0, 1]}, title = "Response Rate"},
    color = {:condition, scale = {range = "#".*hex.(colors)}}) +
@vlplot({:bar, xOffset = 8},
    transform = [{filter = "datum.type == 'falseps'"}],
    x = {:condition, axis = {title = ""}},
    y = {:prop, type = :quantitative, aggregate = :mean},
    color = {value = "#"*hex(neutral)}) +
@vlplot({:rule, xOffset = -8},
    transform = [{filter = "datum.type == 'hits'"}],
    color = {value = "black"},
    x = {:condition, axis = {title = ""}},
    y = {:lower, title = ""}, y2 = :upper
) +
@vlplot({:rule, xOffset = 8},
    transform = [{filter = "datum.type == 'falseps'"}],
    color = {value = "black"},
    x = {:condition, axis = {title = ""}},
    y = {:lower, title = ""}, y2 = :upper
) +
@vlplot({:text, angle = -90, fontSize = 9, align = "right", baseline = "top", dx = 0, dy = 2},
    transform = [{filter = "datum.condition == 'global' && datum.type == 'hits'"}],
    # x = {datum = "spatial"}, y = {datum = 0.},
    x = {:condition, axis = {title = ""}},
    y = {:prop, aggregate = :mean, type = :quantitative},
    text = {value = "Hits"},
) +
@vlplot({:text, angle = -90, fontSize = 9, align = "left", basline = "top", dx = 0, dy = 20},
    transform = [{filter = "datum.condition == 'global' && datum.type == 'falseps'"}],
    # x = {datum = "spatial"}, y = {datum = },
    x = {:condition, axis = {title = ""}},
    y = {datum = 0},
    text = {value = "False Positives"},
) |>
save(joinpath(dir, "raw_behavior.svg"))

means |> @vlplot(
    width = {step = 50},
    config = {
        legend = {disable = true},
        bar = {discreteBandSize = 16}
    }) +
@vlplot({:bar, xOffset = -8},
    transform = [{filter = "datum.type == 'falsetargets'"}],
    x = {:condition, axis = {title = "", labelAngle = 0,
        labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}, },
    y = {:prop, type = :quantitative, aggregate = :mean,
         scale = {domain = [0, 1]}, title = "Response Rate"},
    color = {:condition, scale = {range = "#".*hex.(colors)}}) +
@vlplot({:bar, xOffset = 8},
    transform = [{filter = "datum.type == 'notargets'"}],
    x = {:condition, axis = {title = ""}},
    y = {:prop, type = :quantitative, aggregate = :mean},
    color = {value = "#"*hex(neutral)}) +
@vlplot({:rule, xOffset = -8},
    transform = [{filter = "datum.type == 'falsetargets'"}],
    color = {value = "black"},
    x = {:condition, axis = {title = ""}},
    y = {:lower, title = ""}, y2 = :upper
) +
@vlplot({:rule, xOffset = 8},
    transform = [{filter = "datum.type == 'notargets'"}],
    color = {value = "black"},
    x = {:condition, axis = {title = ""}},
    y = {:lower, title = ""}, y2 = :upper
) +
@vlplot({:text, angle = -90, fontSize = 9, align = "left", baseline = "center", dx = 3, dy = -6},
    transform = [{filter = "datum.condition == 'object' && datum.type == 'falsetargets'"}],
    # x = {datum = "spatial"}, y = {datum = 0.},
    x = {:condition, axis = {title = ""}},
    y = {:upper, aggregate = :mean, type = :quantitative},
    text = {value = "False Targets"},
) +
@vlplot({:text, angle = -90, fontSize = 9, align = "left", basline = "center", dx = 3, dy = 8},
    transform = [{filter = "datum.condition == 'global' && datum.type == 'notargets'"}],
    # x = {datum = "spatial"}, y = {datum = },
    x = {:condition, axis = {title = ""}},
    y = {:upper, aggregate = :mean, type = :quantitative},
    text = {value = "No Targets"},
) |>
save(joinpath(dir, "raw_behavior_falsep.svg"))

# EEG Results
# =================================================================

# Full vs. Null model results
# -----------------------------------------------------------------

file = joinpath(processed_datadir("analyses"), "eeg_condition.csv")
stadata = CSV.read(file, DataFrame)

pl = statdata |>
    @vlplot(
        facet = {column = {field = :comparison, type = :nominal, title = ""}}
    ) +
    (
        @vlplot() +
        @vlplot({:point, filled = true},
            x = {"logitnullmean:q", title = "Null Model Accuracy (Logit Scale)"},
            y = {"logitmean:q", title = "Full Model Accuracy (Logit Scale)"},
        ) +
        (
            @vlplot(data = {values = [{x = -6, y = -6}, {x = 6, y = 6}]}) +
            @vlplot({:line, clip = true, strokeDash = [2 2]},
                color = {value = "black"},
                x = {"x:q", scale = {domain = collect(extrema(statdata.logitnullmean))}},
                y = {"y:q", type = :quantitative, scale = {domain = collect(extrema(statdata.logitmean))}}
            )
        )
    );
pl |> save(joinpath(dir, "category_ind.svg"))

# Different model baselines
# -----------------------------------------------------------------

file = joinpath(processed_datadir("analyses"), "condition-and-baseline.json")
GermanTrack.@load_cache file predictbasedf

baselinedf = @_ predictbasedf |>
    filter(_.hittype == "hit", __) |>
    groupby(__, [:sid, :comparison, :modeltype, :winlen, :hittype]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :correct) |>
    groupby(__, [:sid, :comparison, :modeltype, :hittype]) |>
    combine(__,
        :correct => mean => :mean,
        :correct => logit ∘ shrinktowards(0.5, by=0.01) ∘ mean => :logitmean)

fulldf = @_ baselinedf |> filter(_.modeltype == "full", __) |>
    deletecols!(__, [:modeltype, :mean, :hittype]) |>
    rename!(__, :logitmean => :logitfullmean)

plotdata = @_ baselinedf |>
    filter(_.modeltype != "full", __) |>
    innerjoin(__, fulldf, on = [:sid, :comparison])

pl = plotdata |>
    @vlplot(
        facet = {column = {field = :modeltype, type = :nominal, title = "Baseline"}}
    ) +
    (
        @vlplot() +
        @vlplot({:point, filled = true},
            x = {"logitmean:q", title = "Baseline Model Accuracy (Logit Scale)"},
            y = {"logitfullmean:q", title = "Full Model Accuracy (Logit Scale)"},
            color = :comparison
        ) +
        (
            @vlplot(data = {values = [{x = -6, y = -6}, {x = 6, y = 6}]}) +
            @vlplot({:line, clip = true, strokeDash = [2 2]},
                color = {value = "black"},
                x = {"x:q", scale = {domain = collect(extrema(plotdata.logitmean))}},
                y = {"y:q", type = :quantitative, scale = {domain = collect(extrema(plotdata.logitfullmean))}}
            )
        )
    );
pl |> save(joinpath(dir, "supplement", "category_baselines.svg"))

# TODO: below stuff is old code that may need to be adjusted to work in
# new enviornment

labels = OrderedDict(
    "full" => "Full Model",
    "null" => "Null Model",
    "random-labels" => "Rand.\n Condition",
    "random-trialtype" => "Rand.\n Response",
    "random-window-before" => "Rand.\n Window"
)
tolabel(x) = labels[x]
barwidth = 20
pldata = @_ baselinedf |>
    groupby(__, [:modeltype, :sid]) |>
    @combine(__, mean = mean(:mean)) |>
    @transform(__,
        label = tolabel.(:modeltype),
        isbase = :modeltype .== "full"
    )

plcolors = ColorSchemes.nuuk[[0.85, 0.2]]
pl = pldata |>
    @vlplot(
        config = {
            legend = {disable = true},
            bar = {discreteBandSize = barwidth}},
    ) +
    (
        @vlplot(width = 164, height = 50,
            x = {:label,
                sort = values(labels),
                axis = {title = "", labelAngle = 0,
                    labelExpr = "split(datum.label, '\\n')"}},
            transform = [{filter = "datum.modeltype != 'random-trialtype' &&"*
                " datum.modeltype != 'random-window-before'"}],
            color = {:isbase, type = :nominal, scale = {range = "#".*hex.(plcolors)}},
        ) +
        @vlplot({:bar, clip = true},
            y = {:mean, title = "Accuracy", aggregate = :mean, type = :quantitative,
                scale = {domain = [0.4, 1.0]}},
        ) +
        @vlplot(:errorbar,
            y = {:mean, title = "", aggregate = :ci, type = :quantitative},
            color = {value = "black"}
        )
    );
pl |> save(joinpath(dir, "supplement", "category_baseline_bar_1.svg"))

pl = pldata |>
    @vlplot(
        config = {
            legend = {disable = true},
            bar = {discreteBandSize = barwidth}},
    ) +
    (
        @vlplot(width = 164, height = 50,
            x = {:label,
                sort = values(labels),
                axis = {title = "", labelAngle = 0,
                    labelExpr = "split(datum.label, '\\n')"}},
            transform = [{filter = "datum.modeltype != 'null' &&"*
                " datum.modeltype != 'random-labels'"}],
            color = {:isbase, type = :nominal, scale = {range = "#".*hex.(plcolors)}},
        ) +
        @vlplot({:bar, clip = true},
            y = {:mean, title = "Accuracy", aggregate = :mean, type = :quantitative, scale = {domain = [0.4, 1.0]}},
        ) +
        @vlplot(:errorbar,
            y = {:mean, title = "", aggregate = :ci, type = :quantitative},
            color = {value = "black"}
        )
    );
pl |> save(joinpath(dir, "supplement", "category_baseline_bar_2.svg"))

# Main median power results
# -----------------------------------------------------------------

# Examine the power across bins/channels near a target
# -----------------------------------------------------------------

subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

classhitdf_groups = @_ events |>
    transform!(__, AsTable(:) => ByRow(findresponse) => :hittype) |>
    groupby(__, [:sid, :condition, :hittype])

windows = [(len = 2.0, start = 0.0)]
classhitdf = compute_freqbins(subjects, classhitdf_groups, windowtarget, windows)

chpat = r"channel_([0-9]+)_([a-z]+)"
bincenter(x) = @_ GermanTrack.default_freqbins[Symbol(x)] |> log.(__) |> mean |> exp
classhitdf_long = @_ classhitdf |>
    stack(__, r"channel", [:sid, :condition, :weight, :hittype], variable_name = "channelbin") |>
    transform!(__, :channelbin => ByRow(x -> parse(Int,match(chpat, x)[1])) => :channel) |>
    transform!(__, :channelbin => ByRow(x -> match(chpat, x)[2])            => :bin) |>
    transform!(__, :bin        => ByRow(bincenter)                          => :frequency) |>
    groupby(__, :bin) |>
    transform!(__, :value => zscore => :zvalue)

classhitdf_summary = @_ classhitdf_long |>
    groupby(__, [:hittype, :channel, :frequency, :condition]) |>
    combine(__, :zvalue => median => :medvalue,
                :zvalue => minimum => :minvalue,
                :zvalue => maximum => :maxvalue,
                :zvalue => (x -> quantile(x, 0.75)) => :q50u,
                :zvalue => (x -> quantile(x, 0.25)) => :q50l,
                :zvalue => (x -> quantile(x, 0.975)) => :q95u,
                :zvalue => (x -> quantile(x, 0.025)) => :q95l)

ytitle = "Median Power"
@_ classhitdf_summary |>
    filter(_.channel in 1:5, __) |>
    @vlplot(facet = {
            column = {field = :hittype, type = :ordinal},
            row = {field = :channel, type = :ordinal}
        }) + (
        @vlplot() +
        @vlplot(:line, color = :condition,
            x = {:frequency, scale = {type = :log}}, y = {:medvalue, title = ytitle}) +
        @vlplot(:point, color = :condition,
            x = {:frequency, scale = {type = :log}}, y = {:medvalue, title = ytitle}) +
        @vlplot(:errorband, color = :condition,
            x = {:frequency, scale = {type = :log}}, y = {:q50u, title = ytitle}, y2 = :q50l) +
        @vlplot(:errorbar, color = :condition,
            x = {:frequency, scale = {type = :log}}, y = {:q95u, title = ytitle}, y2 = :q95l)
    )

ytitle = "Median Power"
@_ classhitdf_summary |>
    @vlplot(facet = {
            field = :channel,
            type = :ordinal
        },
        config = {facet = {columns = 10}}
    )+(@vlplot() +
        @vlplot(:point, color = :condition,
            x = :hittype,
            y = {:medvalue, title = ytitle, aggregate = :mean, type = :quantitative}) +
        @vlplot(:errorbar, color = :condition,
            x = :hittype,
            y = {:q95u, title = ytitle, aggregate = :mean, type = :quantitative},
            y2 = :q95l)
    )


classhitdf_summary = @_ classhitdf_long |>
    groupby(__, [:hittype, :condition]) |>
    combine(__, :zvalue => median => :medvalue,
                :zvalue => minimum => :minvalue,
                :zvalue => maximum => :maxvalue,
                :zvalue => (x -> quantile(x, 0.75)) => :q50u,
                :zvalue => (x -> quantile(x, 0.25)) => :q50l,
                :zvalue => (x -> quantile(x, 0.975)) => :q95u,
                :zvalue => (x -> quantile(x, 0.025)) => :q95l)

ytitle = "Median Power"
@_ classhitdf_summary |>
    @vlplot(facet = {
            field = :hittype,
            type = :ordinal
        },
        config = {facet = {columns = 10}}
    )+(@vlplot() +
        @vlplot(:point, color = :condition,
            x = :condition, y = {:medvalue, title = ytitle}) +
        @vlplot(:errorbar, color = :condition,
            x = :condition, y = {:q95u, title = ytitle}, y2 = :q95l)
    )

classhitdf_stats = @_ classhitdf_long |>
    groupby(__, [:hittype, :condition, :sid]) |>
    combine(__, :zvalue => median => :medvalue)

ytitle = "Median Power of top 30 MCCA Components"
plpower = @_ classhitdf_stats |>
    filter(_.hittype == "hit", __) |>
    @vlplot(
        # facet = {
        #     field = :hittype,
        #     type = :ordinal
        # },
        config = {legend = {disable = true}, facet = {columns = 10}}
    )+(@vlplot(height = 300) +
    @vlplot({:point, filled = true, size = 75}, color = :condition,
        x = :condition,
        y = {:medvalue, title = ytitle, type = :quantitative, aggregate = :mean}) +
    @vlplot({:errorbar, ticks = {size = 5}}, color = :condition,
        x = :condition,
        y = {:medvalue, title = ytitle, type = :quantitative, aggregate = :ci}) +
    @vlplot({:point, filled = true, size = 15, opacity = 0.25, xOffset = -5},
        color = {value = "black"},
        x = :condition, y = {:medvalue, title = ytitle})
    ) |> save(joinpath(dir, "medpower.svg"))

ytitle = "Median Power of top 30 MCCA Components"
plpower_hittype = @_ classhitdf_stats |>
    filter(_.hittype != "hit", __) |>
    @vlplot(facet = {
            field = :hittype,
            type = :ordinal
        },
        config = {legend = {disable = true}, facet = {columns = 10}}
    )+(@vlplot(height = 100) +
    @vlplot({:point, filled = true, size = 75}, color = :condition,
        x = :condition,
        y = {:medvalue, title = ytitle, type = :quantitative, aggregate = :mean}) +
    @vlplot({:errorbar, ticks = {size = 5}}, color = :condition,
        x = :condition,
        y = {:medvalue, title = ytitle, type = :quantitative, aggregate = :ci}) +
    @vlplot({:point, filled = true, size = 15, opacity = 0.25, xOffset = -5},
        color = {value = "black"},
        x = :condition, y = {:medvalue, title = ytitle})
    ) |> save(joinpath(dir, "medpower_hittypes.svg"))

# Display of model coefficients
# =================================================================

# Compare coefficients across folds
# -----------------------------------------------------------------

# TODO: nonworking code below this point

centerlen = @_ classdf.winlen |> unique |> sort! |> __[5]
centerstart = @_ classdf.winstart |> unique |> __[argmin(abs.(__ .- 0.0))]

classdf_atlen = @_ classdf |> filter(_.winlen == centerlen && _.winstart == centerstart, __)

classcomps_atlen = [
    "global-v-object"  => @_(classdf_atlen |> filter(_.condition in ["global", "object"],  __)),
    "global-v-spatial" => @_(classdf_atlen |> filter(_.condition in ["global", "spatial"], __)),
    "object-v-spatial" => @_(classdf_atlen |> filter(_.condition in ["object", "spatial"], __))
]

coefdf = mapreduce(append!!, classcomps_atlen) do (comp, data)
    function findclass((key, sdf))
        λ = first(λsid[(sid = first(sdf.sid),)].λ)
        result = testclassifier(LassoClassifier(λ),
            data = sdf, y = :condition, X = r"channel",
            crossval = :sid, n_folds = 10,
            seed = stablehash(:cond_coef_timeline,2019_11_18),
            irls_maxiter = 100, include_model_coefs = true,
            weight = :weight, on_model_exception = :throw)

        result[!, keys(key)] .= permutedims(collect(values(key)))
        result[!, :comparison] .= comp

        result
    end

    groups = pairs(groupby(data, :fold))
    foldxt(append!!, Map(findclass), collect(groups))
end

coefnames_ = pushfirst!(propertynames(coefdf[:,r"channel"]), :C)

coefvals = @_ coefdf |>
    groupby(__, [:label_fold, :fold, :comparison]) |>
    combine(__, coefnames_ .=> (only ∘ unique) .=> coefnames_)

function parsecoef(coef)
    parsed = match(r"channel_([0-9]+)_([a-z]+)",string(coef))
    if !isnothing(parsed)
        chanstr, freqbin = parsed[1], parsed[2]
        chan = parse(Int,chanstr)
        chan, freqbin
    elseif string(coef) == "C"
        missing, missing
    else
        error("Unexpected coefficient $coef")
    end
end

coef_spread = @_ coefvals |>
    stack(__, coefnames_, [:label_fold, :fold, :comparison],
        variable_name = :coef) |>
    transform!(__, :coef => ByRow(x -> parsecoef(x)[1]) => :channel) |>
    transform!(__, :coef => ByRow(x -> parsecoef(x)[2]) => :freqbin)

minabs(x) = x[argmin(abs.(x))]

coef_spread_means = @_ coef_spread |>
    filter(!ismissing(_.channel), __) |>
    groupby(__, [:freqbin, :channel, :comparison]) |>
    combine(__, :value => mean => :value,
                :value => length => :N,
                :value => (x -> quantile(x, 0.75)) => :innerhigh,
                :value => (x -> quantile(x, 0.25)) => :innerlow,
                :value => (x -> quantile(x, 0.95)) => :outerhigh,
                :value => (x -> quantile(x, 0.05)) => :outerlow)

compnames = Dict(
    "global-v-object"  => "Global vs. Object",
    "global-v-spatial" => "Global vs. Spatial",
    "object-v-spatial" => "Object vs. Spatial")

coefmeans_rank = @_ coef_spread_means |>
    groupby(__, [:comparison, :channel]) |>
    combine(__, :value => minimum => :minvalue,
                :outerlow => minimum => :minouter) |>
    sort!(__, [:comparison, :minvalue, :minouter]) |>
    groupby(__, [:comparison]) |>
    transform!(__, :minvalue => (x -> 1:length(x)) => :rank) |>
    innerjoin(coef_spread_means, __, on = [:comparison, :channel]) |>
    transform!(__, :channel => ByRow(x -> string("MCCA ",x)) => :channelstr) |>
    filter(!(_.value == 0 && _.outerlow == 0 && _.outerhigh == 0), __) |>
    transform!(__, :comparison => ByRow(x -> compnames[x]) => :comparisonstr)

ytitle = "Median cross-validated coefficient value"
plcoefs = coefmeans_rank |>
    @vlplot(facet =
        {column = {field = :comparisonstr, title = "Comparison", type = :ordinal}},
        spacing = {column = 45}) +
     (@vlplot(x = {:rank, title = "Coefficient Rank (low-to-high)"},
        color = {:freqbin, type = :ordinal, sort = ["delta","theta","alpha","beta","gamma"],
                 scale = {scheme = "magma"}}) +
      @vlplot({:rule, size = 3}, y = :innerlow, y2 = :innerhigh) +
      @vlplot({:errorbar, size = 1, ticks = {size = 5}, tickSize = 2.5},
        y = {:outerlow, title = ytitle, type = :quantitative}, y2 = "outerhigh:q") +
      @vlplot({:point, filled = true, size = 75},
        y = :value,
        color = {
            field = :freqbin,
            legend = {title = nothing},
            type = :ordinal, sort = ["delta","theta","alpha","beta","gamma"]}) +
      @vlplot(
          transform = [
            {aggregate = [{op = :min, field = :value, as = :min_mvalue}],
             groupby = [:channelstr, :rank]},
            {filter = "(datum.rank <= 3) && (datum.min_mvalue != 0)"}],
          x = :rank,
          mark = {type = :text, align = :left, dx = 5, dy = 5}, text = :channelstr,
          y = {field = :min_mvalue, title = ytitle},
          color = {value = "black"}));

# MCCA visualization
# =================================================================

# plot the features of top two components
# -----------------------------------------------------------------


chpat = r"channel_([0-9]+)_([a-z]+)"
bincenter(x) = @_ GermanTrack.default_freqbins[Symbol(x)] |> log.(__) |> mean |> exp
classdf_long = @_ classdf_atlen |>
    stack(__, r"channel", [:sid, :condition, :weight], variable_name = "channelbin") |>
    transform!(__, :channelbin => ByRow(x -> parse(Int,match(chpat, x)[1])) => :channel) |>
    transform!(__, :channelbin => ByRow(x -> match(chpat, x)[2])            => :bin) |>
    transform!(__, :bin        => ByRow(bincenter)                          => :frequency) |>
    groupby(__, :bin) |>
    transform!(__, :value => zscore => :zvalue)

# plot individual for two best dimensions

function bestchan_features(classdf)
    function (row)
        result = @_ filter(occursin(_.condition,row.comparison) &&
                  _.channel == row.channel &&
                  _.bin == row.freqbin, classdf)
        for col in propertynames(row)
            result[:,col] = row[col]
        end
        result
    end
end

best_channel_df = @_ coef_spread_means |>
    groupby(__, :comparison) |>
    combine(__,
        [:channel, :value] =>
            ((chan,x) -> chan[sortperm(x)[1:2]]) => :channel,
        [:freqbin, :value] =>
            ((bin,x) -> bin[sortperm(x)][1:2]) => :freqbin,
        :channel => (x -> 1:2) => :rank)

classdf_best = @_ foldl(append!!, Map(bestchan_features(classdf_long)),
    eachrow(best_channel_df))

classdf_best_long = @_ classdf_best |>
    unstack(__, [:sid, :condition, :comparison], :rank, :zvalue,
        renamecols = x -> Symbol(:value,x)) |>
    innerjoin(__, unstack(classdf_best, [:sid, :condition, :comparison], :rank, :channelbin,
        renamecols = x -> Symbol(:feat,x)), on = [:sid, :condition, :comparison]) |>
    transform!(__, :comparison => ByRow(x -> compnames[x]) => :comparisonstr) |>
    transform!(__, :feat1 => ByRow(string) => :feat1) |>
    transform!(__, :feat2 => ByRow(string) => :feat2)

function maketitle(x)
    m = match(r"channel_([0-9]+)_([a-z]+)", x)
    chn, bin = m[1], m[2]
    "MCCA Component $chn $(uppercasefirst(bin))"
end
titles = @_ classdf_best_long |>
    groupby(__, :comparison) |>
    combine(__, :feat1 => (maketitle ∘ first) => :title1,
                :feat2 => (maketitle ∘ first) => :title2) |>
    groupby(__, :comparison)

plfeats = @vlplot() + hcat(
    (classdf_best_long |> @vlplot(
        transform = [{filter = "datum.comparison == '$comparison'"}],
        {:point, filled = true},
        x = {:value1, title = titles[(comparison = comparison,)].title1[1],
            scale = {domain = [-2.5, 2.5]}},
        y = {:value2, title = titles[(comparison = comparison,)].title2[1],
            scale = {domain = [-2.5, 2.5]}},
        shape = :condition,
        color = {:condition, scale = {scheme = "dark2"}})
        for comparison in unique(classdf_best_long.comparison))...);

pl = @vlplot(align = "all",
        resolve = {scale = {color = "independent", shape = "independent"}}) +
    vcat(plcoefs, plfeats)

pl |> save(joinpath(dir, "condition_features.svg"))
pl |> save(joinpath(dir, "condition_features.png"))

# Plot spectrum of all components
# -----------------------------------------------------------------

best_channels = skipmissing(best_channel_df.channel) |> unique |> sort!
spectdf_file = joinpath(cache_dir("features"), savename("cond-freaqmeans-spect",
    (channels = best_channels,), "csv", allowedtypes = [Array]))

binsize = 100 / 128
finebins = OrderedDict(string("bin",i) => ((i-1)*binsize, i*binsize) for i in 1:128)
windows = [(len = 2.0, start = 0.0)]

subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")
spectdf_groups = @_ events |>
    filter(_.target_present, __) |>
    filter(findresponse(_) == "hit", __) |>
    groupby(__, [:sid, :condition]);

spectdf = compute_freqbins(subjects, spectdf_groups, windowtarget, windows, foldxt,
    freqbins = finebins, channels = @_ best_channel_df.channel |> unique |> sort! |>
        push!(__, 1, 2))

chpat = r"channel_([0-9]+)_bin([0-9]+)"
spectdf_long = @_ spectdf |>
    stack(__, r"channel", [:sid, :condition, :weight], variable_name = "channelbin") |>
    transform!(__, :channelbin => ByRow(x -> parse(Int,match(chpat, x)[1])) => :channel) |>
    transform!(__, :channelbin => ByRow(x -> parse(Int,match(chpat, x)[2])) => :bin) |>
    transform!(__, :bin => ByRow(bin -> (bin-1)*binsize + binsize/2) => :frequency)

@_ spectdf_long |>
    filter(_.frequency > 3, __) |>
    @vlplot(:line, column = :channel, color = :condition,
        x = {:frequency},
        y = {:value, aggregate = :median, type = :quantitative})

# maybe divide by median value

spectdf_norm = @_ spectdf_long |>
    groupby(__, [:frequency]) |>
    transform!(__, :value => (x -> x ./ median(x)) => :normvalue)

@_ spectdf_norm |>
    filter(_.frequency > 1, __) |>
    @vlplot(facet = {column = {field = :channel}}) +
    (@vlplot() +
        @vlplot(:line, color = :condition,
            x = {:frequency, scale = {type = :log}},
            y = {:normvalue, aggregate = :mean, type = :quantitative}) +
        @vlplot(:errorband, color = :condition,
            x = {:frequency, scale = {type = :log}},
            y = {:normvalue, aggregate = :ci, type = :quantitative}))

# Examine the power across bins/channels near a target
# -----------------------------------------------------------------

subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

classhitdf_groups = @_ events |>
    transform!(__, AsTable(:) => ByRow(findresponse) => :hittype) |>
    groupby(__, [:sid, :condition, :hittype])

windows = [(len = 2.0, start = 0.0)]
classhitdf = compute_freqbins(subjects, classhitdf_groups, windowtarget, windows)

chpat = r"channel_([0-9]+)_([a-z]+)"
bincenter(x) = @_ GermanTrack.default_freqbins[Symbol(x)] |> log.(__) |> mean |> exp
classhitdf_long = @_ classhitdf |>
    stack(__, r"channel", [:sid, :condition, :weight, :hittype], variable_name = "channelbin") |>
    transform!(__, :channelbin => ByRow(x -> parse(Int,match(chpat, x)[1])) => :channel) |>
    transform!(__, :channelbin => ByRow(x -> match(chpat, x)[2])            => :bin) |>
    transform!(__, :bin        => ByRow(bincenter)                          => :frequency) |>
    groupby(__, :bin) |>
    transform!(__, :value => zscore => :zvalue)

classhitdf_summary = @_ classhitdf_long |>
    groupby(__, [:hittype, :channel, :frequency, :condition]) |>
    combine(__, :zvalue => median => :medvalue,
                :zvalue => minimum => :minvalue,
                :zvalue => maximum => :maxvalue,
                :zvalue => (x -> quantile(x, 0.75)) => :q50u,
                :zvalue => (x -> quantile(x, 0.25)) => :q50l,
                :zvalue => (x -> quantile(x, 0.975)) => :q95u,
                :zvalue => (x -> quantile(x, 0.025)) => :q95l)

ytitle = "Median Power"
@_ classhitdf_summary |>
    filter(_.channel in 1:5, __) |>
    @vlplot(facet = {
            column = {field = :hittype, type = :ordinal},
            row = {field = :channel, type = :ordinal}
        }) + (
        @vlplot() +
        @vlplot(:line, color = :condition,
            x = {:frequency, scale = {type = :log}}, y = {:medvalue, title = ytitle}) +
        @vlplot(:point, color = :condition,
            x = {:frequency, scale = {type = :log}}, y = {:medvalue, title = ytitle}) +
        @vlplot(:errorband, color = :condition,
            x = {:frequency, scale = {type = :log}}, y = {:q50u, title = ytitle}, y2 = :q50l) +
        @vlplot(:errorbar, color = :condition,
            x = {:frequency, scale = {type = :log}}, y = {:q95u, title = ytitle}, y2 = :q95l)
    )

ytitle = "Median Power"
@_ classhitdf_summary |>
    @vlplot(facet = {
            field = :channel,
            type = :ordinal
        },
        config = {facet = {columns = 10}}
    )+(@vlplot() +
        @vlplot(:point, color = :condition,
            x = :hittype,
            y = {:medvalue, title = ytitle, aggregate = :mean, type = :quantitative}) +
        @vlplot(:errorbar, color = :condition,
            x = :hittype,
            y = {:q95u, title = ytitle, aggregate = :mean, type = :quantitative},
            y2 = :q95l)
    )


classhitdf_summary = @_ classhitdf_long |>
    groupby(__, [:hittype, :condition]) |>
    combine(__, :zvalue => median => :medvalue,
                :zvalue => minimum => :minvalue,
                :zvalue => maximum => :maxvalue,
                :zvalue => (x -> quantile(x, 0.75)) => :q50u,
                :zvalue => (x -> quantile(x, 0.25)) => :q50l,
                :zvalue => (x -> quantile(x, 0.975)) => :q95u,
                :zvalue => (x -> quantile(x, 0.025)) => :q95l)

ytitle = "Median Power"
@_ classhitdf_summary |>
    @vlplot(facet = {
            field = :hittype,
            type = :ordinal
        },
        config = {facet = {columns = 10}}
    )+(@vlplot() +
        @vlplot(:point, color = :condition,
            x = :condition, y = {:medvalue, title = ytitle}) +
        @vlplot(:errorbar, color = :condition,
            x = :condition, y = {:q95u, title = ytitle}, y2 = :q95l)
    )

classhitdf_stats = @_ classhitdf_long |>
    groupby(__, [:hittype, :condition, :sid]) |>
    combine(__, :zvalue => median => :medvalue)

pl = @_ classhitdf_stats |>
    @vlplot(facet = {
        field = :hittype,
        type = :ordinal
    },
    config = {legend = {disable = true}, facet = {columns = 10}}
    )+(@vlplot(width = 50, height = 300) +
    @vlplot({:point, filled = true, size = 75}, color = :condition,
        x = :condition,
        y = {:medvalue, title = ytitle, type = :quantitative, aggregate = :mean}) +
    @vlplot({:errorbar, ticks = {size = 5}}, color = :condition,
        x = :condition,
        y = {:medvalue, title = ytitle, type = :quantitative, aggregate = :ci}) +
    @vlplot({:point, filled = true, size = 15, opacity = 0.25, xOffset = -5},
        color = {value = "black"},
        x = :condition, y = {:medvalue, title = ytitle})
    )

pl |> save(joinpath(dir, "medpower_hittype.svg"))
pl |> save(joinpath(dir, "medpower_hittype.png"))

# Different channel groups
# =================================================================

# Mean Frequency Bin Analysis
# -----------------------------------------------------------------

classdf_chgroup_file = joinpath(processed_datadir("features"), "cond-freaqmeans-chgroups.csv")

if isfile(classdf_chgroup_file)
    classdf_chgroup = CSV.read(classdf_chgroup_file)
else
    windows = [windowtarget(len = len, start = 0.0)
        for len in GermanTrack.spread(1, 0.5, n_winlens)]

    classdf_chgroup = mapreduce(append!!, ["frontal", "central", "mixed"]) do group
        subjects, events = load_all_subjects(processed_datadir("eeg", group), "h5")
        classdf_chgroup_groups = @_ events |>
            filter(_.target_present, __) |>
            filter(findresponse(_) == "hit", __) |>
            groupby(__, [:sid, :condition])

        result = compute_freqbins(subjects, classdf_chgroup_groups, windows)
        result[!,:chgroup] .= group

        result
    end
    CSV.write(classdf_chgroup_file, classdf_chgroup)
end

# Model evaluation
# -----------------------------------------------------------------

λfold = groupby(final_λs, :fold)
classdf_chgroup[!,:fold] = in.(classdf_chgroup.sid, Ref(Set(λ_folds[1][1]))) .+ 1

classcomps = [
    "global-v-object"  => @_(classdf_chgroup |> filter(_.condition in ["global", "object"],  __)),
    "global-v-spatial" => @_(classdf_chgroup |> filter(_.condition in ["global", "spatial"], __)),
]

resultdf_chgroups = mapreduce(append!!, classcomps) do (comp, data)
    groups = pairs(groupby(data, [:winlen, :fold, :chgroup]))

    progress = Progress(length(groups))
    function findclass((key,sdf))
        result = Empty(DataFrame)
        λ = first(λfold[(fold = first(sdf.fold),)].λ)
        result = testclassifier(LassoPathClassifiers([1.0, λ]), data = sdf, y = :condition,
            X = r"channel", crossval = :sid, n_folds = 10, seed = 2017_09_16,
            weight = :weight, maxncoef = size(sdf[:,r"channel"],2), irls_maxiter = 400,
            on_model_exception = :debug)

        result[!, keys(key)] .= permutedims(collect(values(key)))
        result[!, :comparison] .= comp
        next!(progress)

        result
    end
    foldl(append!!, Map(findclass), collect(groups))
end

# Plot performance
# -----------------------------------------------------------------

classmeans = @_ resultdf_chgroups |>
    groupby(__, [:winlen, :sid, :λ, :fold, :comparison, :chgroup]) |>
    combine(__, [:correct, :weight] => wmean => :mean)

classmeans_sum = @_ classmeans |>
    groupby(__, [:sid, :λ, :fold, :comparison, :chgroup]) |>
    combine(__, :mean => mean => :mean) |>
    transform!(__, :mean => ByRow(logit ∘ shrinktowards(0.5,by=0.01)) => :meanlogit)

nullmeans = @_ classmeans_sum |>
    filter(_.λ == 1.0, __) |>
    rename!(__, :mean => :nullmean, :meanlogit => :nullmeanlogit) |>
    deletecols!(__, :λ)

classdiffs = @_ classmeans_sum |>
    filter(_.λ != 1.0, __) |>
    deletecols!(__, :λ) |>
    innerjoin(__, nullmeans, on = [:comparison, :chgroup, :sid, :fold]) |>
    transform!(__, [:meanlogit, :nullmeanlogit] => (-) => :meandifflogit)

classdiffs |>
    @vlplot(facet = {column = {field = :comparison, type = :nominal}}) + (
        @vlplot(x = {:chgroup, type = :nominal},
            color = {:chgroup, scale = {scheme = :dark2}}) +
        @vlplot(:bar,
            y = {:meandifflogit, aggregate = :mean, type=  :quantitative},
        ) +
        @vlplot(:errorbar,
            color = {value = "black"},
            y = {:meandifflogit, aggregate = :stderr, type=  :quantitative},
        )
    ) |>
    save(joinpath(dir, "chgroups.svg"))

CSV.write(joinpath(processed_datadir("analyses"), "chgroup-accuracy.csv"), classdiffs)
