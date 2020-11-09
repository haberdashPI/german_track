# Setup
# =================================================================

using DrWatson; @quickactivate("german_track")

using GermanTrack, DataFrames, Statistics, Dates, Underscores, Random, Printf,
    ProgressMeter, VegaLite, FileIO, StatsBase, BangBang, Transducers, Infiltrator, Peaks,
    StatsFuns, Distributions, DSP, DataStructures, Colors, Bootstrap, CSV, EEGCoding
wmean = GermanTrack.wmean
n_winlens = 6

dir = mkpath(joinpath(plotsdir(), "condition"))

using GermanTrack: colors, neutral, patterns

# Behavioral Data
# =================================================================

main_effects = CSV.read(joinpath(processed_datadir("plots"), "main_effects.csv"))

barwidth = 20
pl1 = @_ main_effects |>
    filter(_.stat ∈ ["hr", "fr"], __) |>
    transform!(__, [:pmean, :err] => (-) => :lower,
                  [:pmean, :err] => (+) => :upper) |>
    @vlplot(
        width = 242, autosize = "fit",
        config = {
            legend = {disable = true},
            bar = {discreteBandSize = barwidth}
        }) +
    @vlplot({:bar, xOffset = -(barwidth/2)},
        transform = [{filter = "datum.stat == 'hr'"}],
        x = {:condition, axis = {title = "", labelAngle = 0,
            labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}, },
        y = {:pmean, scale = {domain = [0, 1]}, title = "Response Rate"},
        color = {:condition, scale = {range = "#".*hex.(colors)}}) +
    @vlplot({:bar, xOffset = (barwidth/2)},
        transform = [{filter = "datum.stat == 'fr'"}],
        x = {:condition, axis = {title = ""}},
        y = :pmean,
        color = {value = "#"*hex(neutral)}) +
    @vlplot({:rule, xOffset = -(barwidth/2)},
        transform = [{filter = "datum.stat == 'hr'"}],
        color = {value = "black"},
        x = {:condition, axis = {title = ""}},
        y = {:upper, title = ""}, y2 = :lower
    ) +
    @vlplot({:rule, xOffset = (barwidth/2)},
        transform = [{filter = "datum.stat == 'fr'"}],
        color = {value = "black"},
        x = {:condition, axis = {title = ""}},
        y = {:upper, title = ""}, y2 = :lower
    ) +
    @vlplot({:text, angle = -90, fontSize = 9, align = "right", baseline = "bottom", dx = 0, dy = -barwidth-2},
        transform = [{filter = "datum.stat == 'hr' && datum.condition == 'global'"}],
        # x = {datum = "spatial"}, y = {datum = 0.},
        x = {:condition, axis = {title = ""}},
        y = {:pmean, aggregate = :mean, type = :quantitative},
        text = {value = "Hits"},
    ) +
    @vlplot({:text, angle = -90, fontSize = 9, align = "left", baseline = "top", dx = 0, dy = barwidth+2},
        transform = [{filter = "datum.stat == 'fr' && datum.condition == 'global'"}],
        # x = {datum = "spatial"}, y = {datum = },
        x = {:condition, axis = {title = ""}},
        y = {datum = 0},
        text = {value = "False Positives"},
    );
pl1 |> save(joinpath(dir, "behavior.svg"))

pl2 = @_ main_effects |>
    filter(_.stat ∈ ["fr_distract", "fr_random"], __) |>
    transform!(__, [:pmean, :err] => (-) => :lower,
                  [:pmean, :err] => (+) => :upper) |>
    @vlplot(
        width = 242, autosize = "fit",
        config = {
            legend = {disable = true},
            bar = {discreteBandSize = barwidth}
        }) +
    @vlplot({:bar, xOffset = -(barwidth/2)},
        transform = [{filter = "datum.stat == 'fr_distract'"}],
        x = {:condition, axis = {title = "", labelAngle = 0,
            labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}, },
        y = {:pmean, scale = {domain = [0, 1]}, title = "Response Rate"},
        color = {:condition, scale = {range = "#".*hex.(colors[2:3])}}) +
    @vlplot({:bar, xOffset = (barwidth/2)},
        transform = [{filter = "datum.stat == 'fr_random'"}],
        x = {:condition, axis = {title = ""}},
        y = :pmean,
        color = {value = "#"*hex(neutral)}) +
    @vlplot({:rule, xOffset = -(barwidth/2)},
        transform = [{filter = "datum.stat == 'fr_distract'"}],
        color = {value = "black"},
        x = {:condition, axis = {title = ""}},
        y = {:upper, title = ""}, y2 = :lower
    ) +
    @vlplot({:rule, xOffset = (barwidth/2)},
        transform = [{filter = "datum.stat == 'fr_random'"}],
        color = {value = "black"},
        x = {:condition, axis = {title = ""}},
        y = {:upper, title = ""}, y2 = :lower
    ) +
    @vlplot({:text, angle = -90, fontSize = 9, align = "right", baseline = "bottom", dx = 3, dy = -barwidth-2},
        transform = [{filter = "datum.stat == 'fr_distract' && datum.condition == 'object'"}],
        # x = {datum = "spatial"}, y = {datum = 0.},
        x = {:condition, axis = {title = ""}},
        y = {:pmean, aggregate = :mean, type = :quantitative},
        text = {value = "False Target"},
    ) +
    @vlplot({:text, angle = -90, fontSize = 9, align = "left", basline = "top", dx = 0, dy = barwidth+6},
        transform = [{filter = "datum.stat == 'fr_random' && datum.condition == 'object'"}],
        # x = {datum = "spatial"}, y = {datum = },
        x = {:condition, axis = {title = ""}},
        y = {datum = 0},
        text = {value = "No Target"},
    );
pl2 |> save(joinpath(dir, "behavior_distract.svg"))

# Raw Behavioral Experiment Analysis (using merve summaries)
# -----------------------------------------------------------------

summaries = CSV.read(joinpath(processed_datadir("behavioral", "merve_summaries"), "exported_hits.csv"))

ascondition = Dict(
    "test" => "global",
    "feature" => "spatial",
    "object" => "object"
)

rawdata = @_ summaries |>
    transform!(__, :block_type => ByRow(x -> ascondition[x]) => :condition) |>
    rename(__,:sbj_id => :sid) |>
    select(__, :condition, :sid, :hr, :fr, :exp_id) |>
    stack(__, [:hr, :fr], [:condition, :sid, :exp_id],
        variable_name = :type, value_name = :prop)

CSV.write(joinpath(processed_datadir("analyses"), "behavioral_condition.csv"), rawdata)

means = @_ rawdata |>
    groupby(__, [:condition, :type]) |>
    combine(__,
        :prop => mean => :prop,
        :prop => lowerboot => :lower,
        :prop => upperboot => :upper
    )

barwidth = 20
means |> @vlplot(
    width = 242, autosize = "fit",
    # width = {step = 50},
    config = {
        legend = {disable = true},
        bar = {discreteBandSize = barwidth}
    }) +
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
) |>
save(joinpath(dir, "raw_sum_behavior.svg"))

means_byexp = @_ summaries |>
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

# Raw Behavioral Experiment Analysis
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

# Find best λs
# =================================================================

# Mean Frequency Bin Analysis
# -----------------------------------------------------------------

classdf_file = joinpath(processed_datadir("features"), "cond-freaqmeans.csv")

if isfile(classdf_file)
    classdf = CSV.read(classdf_file)
else
    windows = [windowtarget(len = len, start = start)
    for len in 2.0 .^ range(-1, 1, length = 10),
        start in [0; 2.0 .^ range(-2, 2, length = 10)]]

    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")
    classdf_groups = @_ events |>
        filter(_.target_present, __) |>
        filter(ishit(_, region = "target") == "hit", __) |>
        groupby(__, [:sid, :condition])

    classdf = compute_freqbins(subjects, classdf_groups, windows)
    CSV.write(classdf_file, classdf)
end

# Model evaluation
# -----------------------------------------------------------------

λ_folds = folds(2, unique(classdf.sid), rng = stableRNG(2019_11_18, :lambda_folds))
classdf[!,:fold] = in.(classdf.sid, Ref(Set(λ_folds[1][1]))) .+ 1

classcomps = [
    "global-v-object"  => @_(classdf |> filter(_.condition in ["global", "object"],  __)),
    "global-v-spatial" => @_(classdf |> filter(_.condition in ["global", "spatial"], __)),
    "object-v-spatial" => @_(classdf |> filter(_.condition in ["object", "spatial"], __)),
]

lambdas = 10.0 .^ range(-2, 0, length=100)
resultdf = mapreduce(append!!, classcomps) do (comp, data)
    groups = pairs(groupby(data, [:winstart, :winlen, :fold]))

    progress = Progress(length(groups))
    function findclass((key,sdf))
        result = Empty(DataFrame)

        result = testclassifier(LassoPathClassifiers(lambdas), data = sdf, y = :condition,
            X = r"channel", crossval = :sid, n_folds = 10, seed = 2017_09_16,
            weight = :weight, maxncoef = size(sdf[:,r"channel"],2), irls_maxiter = 400,
            on_model_exception = :print)

        result[!, keys(key)] .= permutedims(collect(values(key)))
        result[!, :comparison] .= comp
        next!(progress)

        result
    end
    foldxt(append!!, Map(findclass), collect(groups))
    # foldl(append!!, Map(findclass), collect(groups))
end

# λ selection
# -----------------------------------------------------------------

fold_map, λ_map = pickλ(resultdf, 2, [:comparison, :winlen, :winstart], :comparison,
    smoothing = 0.8, slope_thresh = 0.15, flat_thresh = 0.02, dir = dir)

# Different baseline models
# =================================================================

# Features
# -----------------------------------------------------------------

classbasedf_file = joinpath(cache_dir("features"), savename("baseline-freqmeans",
    (n_winlens = n_winlens, ), "csv"))

if isfile(classbasedf_file)
    classbasedf = CSV.read(classbasedf_file)
else

    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")
    classbasedf_groups = @_ events |>
        transform!(__, AsTable(:) => ByRow(x -> ishit(x, region = "target")) => :hittype) |>
        groupby(__, [:sid, :condition, :hittype])

    baseparams = (mindist = 0.5, minlength = 0.5, onempty = missing)
    windowtypes = [
        "target"    => (;kwds...) -> windowtarget(name = "target"; kwds...)
        "rndbefore" => (;kwds...) -> windowbase_bytarget(>; name = "rndbefore",
            mindist = 0.5, minlength = 0.5, onempty = missing, kwds...)
    ]

    windows = [
        winfn(len = len, start = 0.0) for len in GermanTrack.spread(1, 0.5, n_winlens)
        for (name, winfn) in windowtypes
    ]

    classbasedf = compute_freqbins(subjects, classbasedf_groups, windows)
    CSV.write(classbasedf_file, classbasedf)
    alert("Feature computation complete!")
end

# Classification for different baselines
# -----------------------------------------------------------------

classbasedf[!,:fold] = getindex.(Ref(fold_map), classbasedf.sid)
classbasedf[!,:λ] = getindex.(Ref(λ_map), classbasedf.fold)

modeltype = [
    "full" => (
        filterfn = @_(filter(_.windowtype == "target", __)),
        λfn = df -> df.λ |> first
    ),
    "null" => (
        filterfn = @_(filter(_.windowtype == "target", __)),
        λfn = df -> 1.0,
    ),
    "random-labels" => (
        filterfn = df ->
            @_(df |> filter(_.windowtype == "target", __) |>
                     groupby(__, [:sid, :winlen, :windowtype]) |>
                     transform!(__, :condition => shuffle => :condition)),
        λfn = df -> df.λ |> first
    ),
    "random-window-before" => (
        filterfn = @_(filter(_.windowtype == "rndbefore", __)),
        λfn = df -> df.λ |> first
    ),
    "random-trialtype" => (
        filterfn = df ->
            @_(df |> filter(_.windowtype == "target", __) |>
                     groupby(__, [:sid, :condition, :winlen, :windowtype]) |>
                     transform!(__, :hittype => shuffle => :hittype)),
        λfn = df -> df.λ |> first
    )
]

comparisons = [
    "global-v-object"  => @_(filter(_.condition ∈ ["global", "object"],  __)),
    "global-v-spatial" => @_(filter(_.condition ∈ ["global", "spatial"], __)),
    "object-v-spatial" => @_(filter(_.condition ∈ ["object", "spatial"], __)),
]

function bymodeltype(((key, df_), type, comp))
    df = df_ |> comp[2] |> type[2].filterfn

    result = testclassifier(LassoClassifier(type[2].λfn(df)),
        data = df, y = :condition, X = r"channel",
        crossval = :sid, n_folds = 10,
        seed = stablehash(:cond_baseline,2019_11_18),
        irls_maxiter = 100,
        weight = :weight, on_model_exception = :throw
    )
    if !isempty(result)
        result[!, :modeltype]  .= type[1]
        result[!, :comparison] .= comp[1]
        for (k,v) in pairs(key)
            result[!, k] .= v
        end
    end

    return result
end

predictbasedf = @_ classbasedf |>
    groupby(__, [:fold, :winlen, :hittype]) |> pairs |>
    Iterators.product(__, modeltype, comparisons) |>
    collect |> foldxt(append!!, Map(bymodeltype), __)

# Main classification results
# -----------------------------------------------------------------

predictmeans = @_ predictbasedf |>
    groupby(__, [:sid, :comparison, :modeltype, :winlen, :hittype]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :correct) |>
    groupby(__, [:sid, :comparison, :modeltype, :hittype]) |>
    combine(__,
        :correct => mean => :mean,
        :correct => logit ∘ shrinktowards(0.5, by=0.01) ∘ mean => :logitcorrect)

compnames = OrderedDict(
    "global-v-object"  => "Global vs.\n Object",
    "global-v-spatial" => "Global vs.\n Spatial",
    "object-v-spatial" => "Object vs.\n Spatial")

modelnames = OrderedDict(
    # "random-window" => "Random\nWindow",
    "random-window-before" => "Random\nPre-target\nWindow",
    # "random-window-after" => "Random\nPost-target Window",
    "null" => "Null Model",
    "random-labels" => "Random\nLabels",
    "random-trialtype" => "Random\nTrial Type",
)

Nmtypes = length(modelnames)
refline = DataFrame(x = repeat([0,1],Nmtypes), y = repeat([0,1],Nmtypes),
    modeltype = repeat(keys(modelnames) |> collect, inner=2))
nrefs = size(refline,1)
refline = @_ refline |>
    repeat(__, 3) |>
    insertcols!(__, :comparison => repeat(compnames |> keys |> collect, inner=nrefs))

nullmeans = @_ predictmeans |>
    filter(_.modeltype == "null", __) |>
    deletecols!(__, [:logitcorrect, :modeltype]) |>
    rename!(__, :mean => :nullmean)

nullmean, plotfull =
    let l = logit ∘ shrinktowards(0.5, by = 0.01),
        C = mean(l.(nullmeans.nullmean)),
        tocor = x -> logistic(x + C)


    rawdata = @_ predictmeans |>
        filter(_.modeltype == "full", __) |>
        innerjoin(__, nullmeans, on = [:sid, :comparison, :hittype]) |>
        transform!(__, [:mean, :nullmean] => ByRow((x,y) -> (l(x)-l(y))) => :logitmeandiff) |>
        transform!(__, :mean => ByRow(shrinktowards(0.5, by = 0.01)) => :shrinkmean) |>
        transform!(__, :nullmean => ByRow(l) => :logitnullmean) |>
        transform!(__, :comparison => ByRow(x -> compnames[x]) => :compname)
    CSV.write(joinpath(processed_datadir("analyses"), "eeg_condition.csv"), rawdata)

    logistic(C), @_ rawdata |>
        groupby(__, [:compname, :hittype, :comparison]) |>
        combine(__,
            :logitmeandiff => tocor ∘ mean => :mean,
            :logitmeandiff => (x -> tocor(lowerboot(x, alpha = 0.05))) => :lower,
            :logitmeandiff => (x -> tocor(upperboot(x, alpha = 0.05))) => :upper,
        )
end

ytitle= ["Neural Classification Accuracy", "(Null Model Corrected)"]
barwidth = 25
plhit = @_ plotfull |>
    filter(_.hittype == "hit", __) |>
    @vlplot(
        # facet = { column = { field = :hittype, type = :nominal} },
        width = 242,
        autosize = "fit",
        config = {
            legend = {disable = true},
            bar = {discreteBandSize = barwidth}
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
            scale = {domain = [0 ,1]},
            title = ytitle}) +
    @vlplot({:rule},
        color = {value = "black"},
        y2 = :upper,
        y = {:lower,
            scale = {domain = [0 ,1]},
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
plotfile = joinpath(dir, "category.svg")
plhit |> save(plotfile)
addpatterns(plotfile, patterns, size = 10)

ytitle= "% Correct"
plhittype = @_ plotfull |>
    filter(_.hittype != "hit", __) |>
    @vlplot(
        facet = { column = { field = :hittype, type = :nominal} },
        transform = [{calculate = "datum.correct * 100", as = :correct},
                     {calculate = "datum.nullmodel * 100", as = :nullmodel}],
    ) + (
    @vlplot(x = {:compname, axis = nothing},
        height = 100,
        color = {
            :compname, title = nothing,
            scale = {range = ["url(#mix1_2)", "url(#mix1_3)", "url(#mix2_3)"]},
            legend = {legendX = 5, legendY = 5, orient = "none"}}) +
    @vlplot(:bar,
        y = {:correct, aggregate = :mean, type = :quantitative,
            scale = {domain = [0.5 ,1].*100},
            title = ytitle}) +
    @vlplot(:bar,
        color = {value = "rgb(50,50,50)"},
        opacity = {value = 0.5},
        y = {:nullmodel, aggregate = :mean, type = :quantitative, title = ytitle},
    ) +
    @vlplot({:errorbar, size = 1, ticks = {size = 5}, tickSize = 2.5},
        color = {value = "black"},
        y = {:correct, aggregate = :ci, type = :quantitative,
            scale = {domain = [0.5 ,1].*100},
            title = ytitle})
    );
plotfile = joinpath(dir, "category_hittype.svg")
plhittype |> save(plotfile)
addpatterns(plotfile, patterns)

# Combine above figures into single plot
# -----------------------------------------------------------------

GermanTrack.@usepython

svg = pyimport("svgutils").compose

background_file = tempname()

background = pyimport("svgutils").transform.fromstring("""
    <svg>
        <rect width="100%" height="100%" fill="white"/>
    </svg>
""").save(background_file)

fig = svg.Figure("89mm", "160mm", # "240mm",
    svg.SVG(background_file),
    svg.Panel(
        svg.SVG(joinpath(dir, "raw_sum_behavior.svg")).move(0,15),
        svg.Text("A", 2, 10, size = 12, weight="bold"),
        svg.SVG(joinpath(plotsdir("icons"), "behavior.svg")).
            scale(0.1).move(220,15)
    ).move(0, 0),
    # svg.Panel(
    #     svg.SVG(joinpath(dir, "behavior_distract.svg")).move(0,15),
    #     svg.Text("B", 2, 10, size = 12, weight = "bold")
    # ).move(0, 225),
    svg.Panel(
        svg.SVG(joinpath(dir, "category.svg")).move(0,15),
        svg.Text("B", 2, 10, size = 12, weight = "bold"),
        svg.SVG(joinpath(plotsdir("icons"), "eeg.svg")).
            scale(0.1).move(220,15)
    ).move(0, 225)
        # svg.Text("C", 2, 10, size = 12, weight = "bold")
    # ).move(0, 450)
).scale(1.333).save(joinpath(dir, "fig1.svg"))

# Main median power results
# -----------------------------------------------------------------

# Examine the power across bins/channels near a target
# -----------------------------------------------------------------

subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

classhitdf_groups = @_ events |>
    transform!(__, AsTable(:) => ByRow(x -> ishit(x, region = "target")) => :hittype) |>
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

# Detailed Baseline plots
# -----------------------------------------------------------------

baselines = @_ predictmeans |>
    filter(_.modeltype != "full", __) |>
    deletecols!(__, :correct) |>
    rename!(__, :logitcorrect => :baseline)

plotmeans = @_ predictmeans |>
    filter(_.modeltype == "full", __) |>
    deletecols(__, :modeltype) |>
    innerjoin(__, baselines, on = [:sid, :comparison]) |>
    filter(_.modeltype ∈ keys(modelnames), __) |>
    transform!(__, :comparison => ByRow(x -> compnames[x]) => :compname) |>
    transform!(__, :modeltype => ByRow(x -> modelnames[x]) => :mtypename) |>
    transform!(__, [:logitcorrect, :baseline] => (-) => :correctdiff)

xtitle = "Baseline Accuracy"
ytitle = "Full-model Accuracy"
pl = @vlplot(data = plotmeans,
        facet = {field = :mtypename, title = "Basline Method",
                sort = values(modelnames)},
        config = {facet = {columns = 3}}
    ) + (
        @vlplot() +
        @vlplot(:point,
            color = :comparison,
            y = {:logitcorrect, title = ytitle}, x = {:baseline, title = xtitle}) +
        @vlplot(data = refline, mark = {:line, strokeDash = [2, 2], size = 2},
            y = {:y, title = ytitle},
            x = {:x, title = xtitle},
            color = {value = "black"})
    );
pl |> save(joinpath(dir, "baseline_individual.svg"))

ytitle = "Full Model - Baseline (logit scale)"
pl = plotmeans |>
    @vlplot(
        facet = {
            column = {
                field = :mtypename, sort = collect(values(modelnames)),
                header = {
                    title = "Baseline Method",
                    labelExpr = "split(datum.value, '\\n')"
                }
            }
        }) + (
        @vlplot(x = {:compname, axis = nothing},
            color = {
                :compname, title = nothing,
                scale = {range = ["url(#blue_orange)", "url(#blue_red)", "url(#orange_red)"]},
                legend = {legendX = 5, legendY = 5, orient = "none"}}) +
        @vlplot(:bar,
            y = {:correctdiff, aggregate = :mean, type = :quantitative,
                 title = ytitle}) +
        @vlplot({:errorbar, size = 1, ticks = {size = 5}, tickSize = 2.5},
            color = {value = "black"},
            y = {:correctdiff, aggregate = :ci, type = :quantitative,
                title = ytitle}) +
        @vlplot({:point, filled = true, size = 15, opacity = 0.25, xOffset = -2},
            color = {value = "black"},
            y = :correctdiff)
    );

plotfile = joinpath(dir, "baseline_models.svg")
pl |> save(plotfile)
addpatterns(plotfile, patterns)

# customize the fill with some low-level svg coding

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
    filter(ishit(_, region = "target") == "hit", __) |>
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
    transform!(__, AsTable(:) => ByRow(x -> ishit(x, region = "target")) => :hittype) |>
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
            filter(ishit(_, region = "target") == "hit", __) |>
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



