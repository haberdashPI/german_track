# Setup
# =================================================================

using DrWatson; @quickactivate("german_track")

using GermanTrack, DataFrames, Statistics, Dates, Underscores, Random, Printf,
    ProgressMeter, VegaLite, FileIO, StatsBase, BangBang, Transducers, Infiltrator, Peaks,
    StatsFuns, Distributions, DSP, DataStructures, Colors, Bootstrap, CSV, EEGCoding
wmean = GermanTrack.wmean
n_winlens = 6

dir = mkpath(joinpath(plotsdir(), "condition", "supplement"))

using GermanTrack: colors, neutral, patterns

# Behavioral Analyses
# =================================================================

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
        ByRow(x -> ishit(x, region = "target", mark_false_targets = true)) => :hittype) |>
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

# Behavioral data from EEG Experiment
# -----------------------------------------------------------------

subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

means = @_ events |>
    transform!(__, AsTable(:) =>
        ByRow(x -> ishit(x, region = "target", mark_false_targets = true)) => :hittype) |>
    groupby(__, [:condition, :sid]) |>
    combine(__, :hittype => (x -> mean(==("hit"), x)) => :hits,
                :hittype => (x -> mean(y -> occursin("falsep", y), x)) => :falseps,
                :hittype => (x -> mean(==("falsep"), x)) => :notargets,
                :hittype => (x -> mean(==("falsep-target"), x)) => :falsetargets) |>
    stack(__, [:hits, :falseps, :notargets, :falsetargets], [:condition, :sid],
        variable_name = :type, value_name = :proportion) |>
    groupby(__, [:condition, :type]) |>
    combine(__, :proportion => mean => :prop,
                :proportion => lowerboot => :lower,
                :proportion => upperboot => :upper)


means |> @vlplot(
    width = {step = 50},
    config = {
        legend = {disable = true},
        bar = {discreteBandSize = 16}
    }) +
@vlplot({:bar, xOffset = -8},
    transform = [{filter = "datum.type == 'hits'"}],
    x = {:condition, axis = {title = "", labelAngle = -24,
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
) |>
#  +
# @vlplot({:text, angle = -90, fontSize = 9, align = "right", baseline = "top", dx = 0, dy = 9},
#     transform = [{filter = "datum.stat == 'hr'"}],
#     # x = {datum = "spatial"}, y = {datum = 0.},
#     x = {:condition, axis = {title = ""}},
#     y = {:pmean, aggregate = :mean, type = :quantitative},
#     text = {value = "Hits"},
# ) +
# @vlplot({:text, angle = -90, fontSize = 9, align = "left", basline = "top", dx = 0, dy = 13},
#     transform = [{filter = "datum.stat == 'fr'"}],
#     # x = {datum = "spatial"}, y = {datum = },
#     x = {:condition, axis = {title = ""}},
#     y = {datum = 0},
#     text = {value = "False Positives"},
# ) |>
save(joinpath(dir, "eeg_behavior.svg"))

function dprime(hits,falarm)
    quantile(Normal(),hits) - quantile(Normal(),falarm)
end

dprimes = @_ events |>
    transform!(__, AsTable(:) =>
        ByRow(x -> ishit(x, region = "target", mark_false_targets = true)) => :hittype) |>
    groupby(__, [:condition, :sid]) |>
    combine(__, :hittype =>
        (x -> dprime(mean(==("hit"), x), mean(y -> occursin("falsep",y), x))) => :dprime)

dprimes |> @vlplot() +
    @vlplot(:point,
        x = {:condition, axis = {title = "", labelAngle = -24,
            labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}, },
        y = {:dprime, type = :quantitative, aggregate = :mean, title = "d'"}) +
    @vlplot(:errorbar,
        x = {:condition, axis = {title = "", labelAngle = -24,
            labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}, },
        y = {:dprime, type = :quantitative, aggregate = :ci, title = "d'"}) |>
save(joinpath(dir, "eeg_dprime.svg"))

dprimes_t = @_ events |>
    transform!(__, AsTable(:) =>
        ByRow(x -> ishit(x, region = "target", mark_false_targets = true)) => :hittype) |>
    groupby(__, [:condition, :sid]) |>
    combine(__, :hittype =>
        (x -> dprime(mean(y -> y in ["hit", "falsep-target"], x), mean(==("falsep"), x))) => :dprime)

dprimes_t |> @vlplot() +
    @vlplot(:point,
        x = {:condition, axis = {title = "", labelAngle = -24,
            labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}, },
        y = {:dprime, type = :quantitative, aggregate = :mean, title = "d'"},
       ) +
    @vlplot(:errorbar,
        x = {:condition, axis = {title = "", labelAngle = -24,
            labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}, },
        y = {:dprime, type = :quantitative, aggregate = :ci, title = "d'"}) |>
save(joinpath(dir, "eeg_dprime_falsetarget.svg"))

