# Setup
# =================================================================

n_winlens = 12
n_folds = 10

using DrWatson; @quickactivate("german_track")
using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures, Dates, Underscores,
    Printf, ProgressMeter, VegaLite, FileIO, StatsBase, BangBang, Transducers,
    Infiltrator, Peaks, Distributions, DSP, Random, CategoricalArrays, StatsModels,
    StatsFuns, Colors, CSV

using GermanTrack: colors, gray, patterns, lightdark, darkgray, seqpatterns

dir = mkpath(plotsdir("figure3_parts"))

# Hit rate x salience level (Figure 3A-B)
# =================================================================

summaries = CSV.read(joinpath(processed_datadir("behavioral", "merve_summaries"),
    "export_salience.csv"), DataFrame)

ascondition = Dict(
    "test" => "global",
    "feature" => "spatial",
    "object" => "object"
)

indmeans = @_ summaries |>
    transform!(__, :block_type => ByRow(x -> ascondition[x]) => :condition) |>
    rename(__,:sbj_id => :sid, :hr_highsal => :high, :hr_lowsal => :low) |>
    select(__, :condition, :sid, :high, :low) |>
    stack(__, [:high, :low], [:condition, :sid],
        variable_name = :salience_label, value_name = :prop)

CSV.write(joinpath(processed_datadir("analyses"), "behavior_salience.csv"), indmeans)

# Absolute values (Figure 3A)
# -----------------------------------------------------------------

means = @_ indmeans |>
    groupby(__, [:condition, :salience_label]) |>
    combine(__,
        :prop => mean => :mean,
        :prop => lowerboot => :lower,
        :prop => upperboot => :upper
    ) |>
    transform(__, [:condition, :salience_label] => ByRow(string) => :condition_salience)

barwidth = 18
ytitle = "Hit Rate"
yrange = [0, 1]
pl = means |>
    @vlplot(
        height = 175, width = 242, autosize = "fit",
        config = {
            legend = {disable = true},
            bar = {discreteBandSize = barwidth},
            axis = {titlePadding = 13}
        },
    ) +
    @vlplot({:bar, xOffset = -(barwidth/2), clip = true},
        transform = [{filter = "datum.salience_label == 'high'"}],
        color = {:condition_salience, title = nothing, scale = {range = "#".*hex.(lightdark)}},
        x = {:condition, axis = {title = "", labelAngle = 0,
            labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}},
        y = {:mean, title = ytitle, scale = {domain = yrange}}
    ) +
    @vlplot({:rule, xOffset = -(barwidth/2)},
        transform = [{filter = "datum.salience_label == 'high'"}],
        color = {value = "black"},
        x = {:condition, title = nothing},
        y = {:lower, title = ""}, y2 = :upper
    ) +
    @vlplot({:bar, xOffset = (barwidth/2), clip = true},
        transform = [{filter = "datum.salience_label == 'low'"}],
        color = {:condition_salience, title = nothing},
        x = {:condition, title = nothing},
        y = {:mean, title = ytitle}
    ) +
    @vlplot({:rule, xOffset = (barwidth/2)},
        transform = [{filter = "datum.salience_label == 'low'"}],
        x = {:condition, title = nothing},
        color = {value = "black"},
        y = {:lower, title = ""}, y2 = :upper
    ) +
    @vlplot({:text, angle = -90, fontSize = 9, align = "left", baseline = "bottom", dx = 0, dy = -barwidth-2},
        transform = [{filter = "datum.salience_label == 'high' && datum.condition == 'global'"}],
        # x = {datum = "spatial"}, y = {datum = 0.},
        x = {:condition, axis = {title = ""}},
        y = {datum = yrange[1]},
        text = {value = "High Salience"},
    ) +
    @vlplot({:text, angle = -90, fontSize = 9, align = "left", baseline = "top", dx = 0, dy = barwidth+2},
        transform = [{filter = "datum.salience_label == 'low' && datum.condition == 'global'"}],
        # x = {datum = "spatial"}, y = {datum = },
        x = {:condition, axis = {title = ""}},
        y = {datum = yrange[1]},
        text = {value = "Low Salience"},
    ); # +
    # (
    #     @vlplot(data = {values = [{}]}) +
    #     @vlplot({:text, angle = 0, fontSize = 9, align = "left", baseline = "line-top",
    #         dx = -2barwidth - 17, dy = 22},
    #         color = {value = "#"*hex(darkgray)},
    #         x = {datum = "global"},
    #         y = {datum = yrange[1]},
    #         text = {datum = ["Less distinct", "response during switch"]}
    #     ) +
    #     @vlplot({:text, angle = 0, fontSize = 9, align = "left", baseline = "line-bottom",
    #         dx = -2barwidth - 17, dy = -24},
    #         color = {value = "#"*hex(darkgray)},
    #         x = {datum = "global"},
    #         y = {datum = yrange[2]},
    #         text = {datum = ["More distinct", "response during switch"]}
    #     )
    # ) +
pl |> save(joinpath(dir, "fig3a.svg"))

# Differences (Figure 3B)
# -----------------------------------------------------------------

diffmeans = @_ indmeans |>
    unstack(__, [:condition, :sid], :salience_label, :prop) |>
    transform!(__, [:high, :low] => (-) => :meandiff) |>
    groupby(__, :condition) |>
    combine(__,
        :meandiff => mean => :meandiff,
        :meandiff => lowerboot => :lower,
        :meandiff => upperboot => :upper
    )

barwidth = 18
pl = @_ diffmeans |>
    @vlplot(
        width = 111, autosize = "fit",
        config = {
            legend = {disable = true},
            bar = {discreteBandSize = barwidth}
        }
    ) +
    @vlplot(:bar,
        x = {:condition,
            title = "",
            axis = {labelAngle = -32, labelAlign = "right",
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}
        },
        color = {:condition, scale = {
            range = urlcol.(keys(seqpatterns))}},
        y = {:meandiff,
            title = "High - Low Salience (Hit Rate)"
        }
    ) +
    @vlplot(:errorbar,
        x = {:condition, title = ""},
        y = {:lower, title = ""}, y2 = :upper,
    );
plotfile =  joinpath(dir, "fig3b.svg")
pl |> save(plotfile)
addpatterns(plotfile, seqpatterns, size = 10)

# Find hyperparameters (λ and winlen)
# =================================================================

file = joinpath(processed_datadir("analyses"), "salience-hyperparams.json")
GermanTrack.@cache_results file fold_map λ_map winlen_map begin
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")
    lambdas = 10.0 .^ range(-2, 0, length=100)

    classdf = compute_freqbins(
        subjects = subjects,
        groupdf  = @_( events |> filter(ishit(_) == "hit", __) |>
            groupby(__, [:sid, :condition, :salience_label])),
        windows  = [windowtarget(len = len, start = start)
            for len in 2.0 .^ range(-1, 1, length = 10),
                start in [0; 2.0 .^ range(-2, 2, length = 10)]]
    )

    resultdf = @_ classdf |>
        addfold!(__, 2, :sid, rng = stableRNG(2019_11_18, :salience_hyper_folds)) |>
        mapgroups(__, desc = "Evaluating hyperparameters...",
            [:winstart, :winlen, :fold, :condition],
            function(sdf)
                testclassifier(LassoPathClassifiers(lambdas),
                    data         = sdf,
                    y            = :salience_label,
                    X            = r"channel",
                    crossval     = :sid,
                    n_folds      = 10,
                    seed         = 2017_09_16,
                    weight       = :weight,
                    maxncoef     = size(sdf[:, r"channel"], 2),
                    irls_maxiter = 600,
                )
            end)

    fold_map = @_ resultdf |>
        groupby(__, :sid) |> combine(__, :fold => first => :fold) |>
        Dict(row.sid => row.fold for row in eachrow(__))

    λ_map, winlen_map = pick_λ_winlen(resultdf,
        [:condition, :sid, :winstart], :condition,
        smoothing = 0.75, slope_thresh = 0.15, flat_thresh = 0.01,
        dir = mkpath(joinpath(dir, "supplement")))

    @info "Saving plots to $(joinpath(dir, "supplement"))"
end

# Plot timeline
# =================================================================

# Compute frequency bins
# -----------------------------------------------------------------

file = joinpath(cache_dir("features"), "salience-freqmeans-timeline.json")
GermanTrack.@cache_results file restuldf_timeline begin
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
            for start in range(0, 3, length = 64) for k in 1:n_winlens
        ]
    )

    resultdf_timeline = @_ classdf |>
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

# Display classification timeline
# -----------------------------------------------------------------

classmeans = @_ resultdf_timeline |>
    groupby(__, [:winstart, :winlen, :sid, :λ, :fold, :condition, :hittype]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :mean)

classmeans_sum = @_ classmeans |>
    groupby(__, [:winstart, :sid, :λ, :fold, :condition, :hittype]) |>
    combine(__, :mean => mean => :mean)

nullmeans = @_ classmeans_sum |>
    filter(_.λ == 1.0, __) |>
    rename!(__, :mean => :nullmean) |>
    deletecols!(__, :λ)

nullmean, classdiffs = let l = logit ∘ shrinktowards(0.5, by = 0.01), C = mean(l.(nullmeans.nullmean))
    100logistic(C),
    @_ classmeans_sum |>
        innerjoin(__, nullmeans, on = [:winstart, :condition, :sid, :fold, :hittype]) |>
        filter(_.λ != 1.0, __) |>
        transform!(__, [:mean, :nullmean] => ByRow((x,y) -> l(x) - l(y)) => :logitmeandiff)
end

annotate = @_ map(abs(_ - 3.0), classdiffs.winstart) |> classdiffs.winstart[argmin(__)]
ytitle = "Full - Null Model (logit scale)"
target_len_y = 0.1
pl = classdiffs |>
    @vlplot(
        config = {legend = {disable = true}},
        title = "Salience Classification Accuracy from EEG",
        facet = {column = {field = :hittype, type = :nominal}}) +
    (@vlplot(
        color = {field = :condition, type = :nominal},
    ) +
    # data lines
    @vlplot(:line,
        x = {:winstart, type = :quantitative, title = "Time (s) Relative to Target Onset"},
        y = {:logitmeandiff, aggregate = :mean, type = :quantitative, title = ytitle,
            #= scale = {domain = [50,67]} =#}) +
    # data errorbands
    @vlplot(:errorband,
        x = {:winstart, type = :quantitative, title = "Time (s) Relative to Target Onset"},
        y = {:logitmeandiff, aggregate = :ci, type = :quantitative, title = ytitle}) +
    # condition labels
    @vlplot({:text, align = :left, dx = 5},
        transform = [{filter = "datum.winstart > 2.2"}],
        x = {datum = 3.0},
        y = {:logitmeandiff, aggregate = :mean, type = :quantitative},
        text = :condition
    ) +
    # Basline (0 %) dotted line
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:rule, strokeDash = [4 4], size = 2},
            y = {datum = 0.0},
            color = {value = "black"})
    ) +
    # "Null Model" text annotation
    # (
    #     @vlplot(data = {values = [{}]}) +
    #     @vlplot(mark = {:text, size = 11, baseline = "line-top", dy = 4},
    #         x = {datum = 2.3}, y = {datum = nullmean},
    #         text = {value = ["Mean Accuracy", "of Null Model"]},
    #         color = {value = "black"}
    #     )
    # ) +
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
        @vlplot(mark = {:text, size = 11, baseline = "bottom", yOffset = -3},
            x = {datum = 0.5}, y = {datum = target_len_y},
            text = {value = "Target Length"},
            color = {value = "black"}
        )
    ));
pl |> save(joinpath(dir, "salience_timeline_hitmiss.svg"))

# Job app plot
# -----------------------------------------------------------------

nullmean, classdiffs =
let l = logit ∘ shrinktowards(0.5, by = 0.01),
    C = mean(l.(nullmeans.nullmean))
    logistic(C),
    @_ classmeans_sum |>
        innerjoin(__, nullmeans, on = [:winstart, :condition, :sid, :fold, :hittype]) |>
        filter(_.λ != 1.0, __) |>
        transform!(__, :condition => ByRow(uppercasefirst) => :condition_label) |>
        transform!(__, [:mean, :nullmean] =>
            ByRow((x,y) -> logistic(l(x) - l(y) + C)) => :meancor) |>
        transform!(__, [:mean, :nullmean] =>
            ByRow((x,y) -> l(x) - l(y)) => :logitmeandiff) |>
        transform!(__, :mean => ByRow(shrinktowards(0.5, by = 0.01)) => :shrinkmean) |>
        transform!(__, :nullmean => ByRow(l) => :logitnullmean)
end

timeslice = 2.8

CSV.write(processed_datadir("analyses", "eeg_salience_timeline.csv"), classdiffs)

annotate = @_ map(abs(_ - 3.0), classdiffs.winstart) |> classdiffs.winstart[argmin(__)]
ytitle = ["Salience Classification (High/Low)", "Accuracy (Null Model Corrected)"]
target_len_y = 0.8
pl = @_ classdiffs |>
    filter(_.hittype == "hit", __) |>
    @vlplot(
        width = 242, height = 200, autosize = "fit",
        config = {legend = {disable = true}},
    ) +
    (@vlplot(
        color = {field = :condition, type = :nominal, scale = {range = "#".*hex.(colors)}},
    ) +
    # data lines
    @vlplot({:line, strokeCap = :round},
        strokeDash = {:condition, type = :nominal, scale = {range = [[1, 0], [6, 4], [2, 4]]}},
        x = {:winstart, type = :quantitative, title = "Time relative to target onset (s)"},
        y = {:meancor, aggregate = :mean, type = :quantitative, title = ytitle,
            scale = {domain = [0.5,1.0]}}) +
    # data errorbands
    @vlplot(:errorband,
        x = {:winstart, type = :quantitative},
        y = {:meancor, aggregate = :ci, type = :quantitative, title = ytitle}) +
    # condition labels
    @vlplot({:text, align = :left, dx = 5},
        transform =
            [{filter = "(datum.condition != 'spatial' && "*
                "datum.winstart > 2.95 && datum.winstart <= 3.0) ||"*
                "(datum.condition == 'spatial' && "*
                "datum.winstart > 2.45 && datum.winstart <= 2.55)"}],
        x = {datum = 3.0},
        y = {:meancor, aggregate = :mean, type = :quantitative},
        text = :condition_label
    ) +
    # "Time Slice" annotation
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot({:rule, strokeDash = [2 2]},
            x = {datum = timeslice},
            color = {value = "black"}
        ) +
        @vlplot({:text, align = "right", dx = -2},
            x = {datum = timeslice},
            y = {datum = 0.98},
            text = {value = ["panel C", "time slice"]},
            color = {value = "black"}
        )
    ) +
    # "Null Model" text annotation
    (
        @vlplot(data = {values = [{}]}) +
        # white rectangle to give text a background
        @vlplot(:rect,
            x = {datum = 2.5}, x2 = {datum = 3},
            y = {datum = 0.5}, y2 = {datum = nullmean},
            color = {value = "white"},
            opacity = {value = 1.0}
        ) +
        @vlplot(mark = {:text, size = 11, baseline = "line-top", dy = 2, align = "center"},
            x = {datum = 3}, y = {datum = nullmean},
            text = {value = ["Null Model", "Accuracy"]},
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
pl |> save(joinpath(dir, "fig3d.svg"))

# Salience class accuracy at fixed time point
# -----------------------------------------------------------------

times = classdiffs.winstart |> unique
real_timeslice = times[argmin(abs.(times .- timeslice))]
barwidth = 16

ytitle = ["Salience Classification (High/Low)", "Accuracy (Null Model Corrected)"]
pl = @_ classdiffs |>
    filter(_.hittype == "hit", __) |>
    filter(_.winstart == real_timeslice, __) |>
    groupby(__, [:condition]) |>
    combine(__,
        :meancor => mean => :meancor,
        :meancor => lowerboot => :lower,
        :meancor => upperboot => :upper,
    ) |>
    @vlplot(
        width = 111, autosize = "fit",
        config = {
            legend = {disable = true},
            bar = {discreteBandSize = barwidth}
        }
    ) +
    @vlplot(:bar,
        x = {:condition,
            title = "",
            axis = {labelAngle = -32, labelAlign = "right",
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}
        },
        color = {:condition, scale = {range = urlcol.(keys(seqpatterns))}},
        y = {:meancor,
            title = ytitle,
            scale = {domain = [0.5, 1.0]}
        }
    ) +
    @vlplot(:errorbar,
        x = {:condition, title = ""},
        y = {:lower, title = ""}, y2 = :upper,
    ) +
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:rule, strokeDash = [4 4], size = 2},
            y = {datum = nullmean},
            color = {value = "black"})
    )
plotfile = joinpath(dir, "fig3c.svg")
pl |> save(plotfile)
addpatterns(plotfile, seqpatterns, size = 10)

# Final, combined plots for data fig 2
# -----------------------------------------------------------------

GermanTrack.@usepython

svg = pyimport("svgutils").compose

background_file = tempname()

background = pyimport("svgutils").transform.fromstring("""
    <svg>
        <rect width="100%" height="100%" fill="white"/>
    </svg>
""").save(background_file)

fig = svg.Figure("89mm", "240mm",
    svg.SVG(background_file),
    svg.Panel(
        svg.SVG(joinpath(dir, "fig3a.svg")).move(0,15),
        svg.Text("A", 2, 10, size = 12, weight="bold"),
        svg.SVG(joinpath(plotsdir("icons"), "behavior.svg")).
            scale(0.1).move(220,15)
    ).move(0, 0),
    svg.Panel(
        svg.SVG(joinpath(dir, "fig3b.svg")).move(0,15),
        svg.Text("B", 2, 10, size = 12, weight="bold"),
        svg.SVG(joinpath(plotsdir("icons"), "behavior.svg")).
            scale(0.1).move(90,15)
    ).move(0, 225),
    svg.Panel(
        svg.SVG(joinpath(dir, "fig3c.svg")).move(0,15),
        svg.Text("C", 2, 10, size = 12, weight = "bold"),
        svg.SVG(joinpath(plotsdir("icons"), "eeg.svg")).
            scale(0.1).move(90,15)
    ).move(125, 225),
    svg.Panel(
        svg.SVG(joinpath(dir, "fig3d.svg")).move(0,15),
        svg.Text("D", 2, 10, size = 12, weight = "bold"),
        svg.SVG(joinpath(plotsdir("icons"), "eeg.svg")).
            scale(0.1).move(215,15)
    ).move(0, 450)
).scale(1.333).save(joinpath(dir, "fig3.svg"))

# hit miss plot
# -----------------------------------------------------------------

hitvmiss = @_ classdiffs |>
    unstack(__, [:winstart, :sid, :condition], :hittype, :logitmeandiff) |>
    transform!(__, [:hit, :miss] => (-) => :hitvmiss)

annotate = @_ map(abs(_ - 3.0), classdiffs.winstart) |> classdiffs.winstart[argmin(__)]
ytitle = "Hit - Miss (logit scale)"
pl = hitvmiss |>
    @vlplot(
        config = {legend = {disable = true}},
        title = "Low/High Salience Classification Accuracy",
        # facet = {column = {field = :hittype, type = :nominal}}
    ) +
    (@vlplot(
        color = {field = :condition, type = :nominal},
    ) +
    # data lines
    @vlplot(:line,
        x = {:winstart, type = :quantitative, title = "Time (s) Relative to Target Onset"},
        y = {:hitvmiss, aggregate = :mean, type = :quantitative, title = ytitle}) +
    # data errorbands
    @vlplot(:errorband,
        x = {:winstart, type = :quantitative, title = "Time (s) Relative to Target Onset"},
        y = {:hitvmiss, aggregate = :ci, type = :quantitative, title = ytitle}) +
    # condition labels
    @vlplot({:text, align = :left, dx = 5},
        transform = [{filter = "datum.winstart > 2.2"}],
        x = {datum = 3.0},
        y = {:hitvmiss, aggregate = :mean, type = :quantitative},
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
            {x = 0.05, y = 0.25, dir = 270},
            {x = 0.95, y = 0.25, dir = 90}]}) +
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
            x = {datum = 0.5}, y = {datum = 0.25},
            text = {value = "Target Length"},
            color = {value = "black"}
        )
    ));
pl |> save(joinpath(dir, "salience_timeline_hitvmiss.svg"))

# Frontal vs. Central Salience Timeline
# =================================================================

# Compute frequency bins
# -----------------------------------------------------------------

classdf_chgroup_file = joinpath(cache_dir("features"), "salience-freqmeans-timeline-chgroups.csv")

if isfile(classdf_chgroup_file)
    classdf_chgroup = CSV.read(classdf_chgroup_file)
else
    function chgroup_freqmeans(group)
        subjects, events = load_all_subjects(processed_datadir("eeg", group), "h5")

        classdf_chgroup_groups = @_ events |>
            transform!(__, AsTable(:) => ByRow(x -> ishit(x, region = "target")) => :hittype) |>
            filter(_.hittype ∈ ["hit", "miss"], __) |>
            groupby(__, [:sid, :condition, :salience_label, :hittype])

        windows = [
            windowtarget(windowfn = event -> (
                start = start,
                len = winlen_bysid(event.sid[1]) |> GermanTrack.spread(0.5,n_winlens,indices=k)))
            for start in range(0, 3, length = 64) for k in 1:n_winlens
        ]
        result = compute_freqbins(subjects, classdf_chgroup_groups, windows)
        result[!, :chgroup] .= group

        result
    end
    classdf_chgroup = foldl(append!!, Map(chgroup_freqmeans), ["frontal", "central", "mixed"])

    CSV.write(classdf_chgroup_file, classdf_chgroup)
end

# Compute classification accuracy
# -----------------------------------------------------------------

resultdf_chgroup_file = joinpath(cache_dir("models"), "salience-chgroup.csv")
classdf_chgroup[!,:fold] = getindex.(Ref(fold_map), classdf_chgroup)

if isfile(resultdf_chgroup_file) && mtime(resultdf_chgroup_file) > mtime(classdf_chgroup_file)
    resultdf_chgroup = CSV.read(resultdf_chgroup_file)
else
    factors = [:fold, :winlen, :winstart, :condition, :hittype, :chgroup]
    groups = groupby(classdf_chgroup, factors)

    progress = Progress(length(groups))
    function findclass((key, sdf))
        λ = λ_map[first(sdf.fold)]
        result = testclassifier(LassoPathClassifiers([1.0, λ]),
            data = sdf, y = :salience_label, X = r"channel", crossval = :sid,
            n_folds = n_folds, seed = stablehash(:salience_classification, :chgroup, 2019_11_18),
            maxncoef = size(sdf[:,r"channel"], 2),
            irls_maxiter = 600, weight = :weight, on_model_exception = :throw)
        result[!, keys(key)] .= permutedims(collect(values(key)))
        next!(progress)

        result
    end

    resultdf_chgroup = @_ groups |> pairs |> collect |>
        # foldl(append!!, Map(findclass), __)
        foldxt(append!!, Map(findclass), __)

    ProgressMeter.finish!(progress)
    CSV.write(resultdf_chgroup_file, resultdf_chgroup)

    alert("Completed salience timeline classification!")
end

# Timeline x Channel group
# -----------------------------------------------------------------

# Display classification timeline
# -----------------------------------------------------------------

classmeans = @_ resultdf_chgroup |>
    groupby(__, [:winstart, :winlen, :sid, :λ, :fold, :condition, :hittype, :chgroup]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :mean)

classmeans_sum = @_ classmeans |>
    groupby(__, [:winstart, :sid, :λ, :fold, :condition, :hittype, :chgroup]) |>
    combine(__, :mean => mean => :mean)

nullmeans = @_ classmeans_sum |>
    filter(_.λ == 1.0, __) |>
    rename!(__, :mean => :nullmean) |>
    deletecols!(__, :λ)

nullmean, classdiffs = let l = logit ∘ shrinktowards(0.5, by = 0.01), C = mean(l.(nullmeans.nullmean))
    100logistic(C),
    @_ classmeans_sum |>
        innerjoin(__, nullmeans, on = [:winstart, :condition, :sid, :fold, :hittype, :chgroup]) |>
        filter(_.λ != 1.0, __) |>
        transform!(__, [:mean, :nullmean] => ByRow((x,y) -> l(x) - l(y)) => :logitmeandiff)
end

annotate = @_ map(abs(_ - 3.0), classdiffs.winstart) |> classdiffs.winstart[argmin(__)]
ytitle = "Full - Null Model (logit scale)"
target_len_y = 1.0
pl = classdiffs |>
    @vlplot(
        config = {legend = {disable = true}},
        title = {text = "Salience Classification Accuracy", subtitle = "Object Condition"},
        facet = {column = {field = :hittype, type = :nominal}}) +
    (@vlplot(
        color = {field = :chgroup, type = :nominal, scale = {scheme = "set2"}},
    ) +
    # data lines
    @vlplot(:line,
        x = {:winstart, type = :quantitative, title = "Time (s) Relative to Target Onset"},
        y = {:logitmeandiff, aggregate = :mean, type = :quantitative, title = ytitle,
            #= scale = {domain = [50,67]} =#}) +
    # data errorbands
    @vlplot(:errorband,
        x = {:winstart, type = :quantitative, title = "Time (s) Relative to Target Onset"},
        y = {:logitmeandiff, aggregate = :ci, type = :quantitative, title = ytitle}) +
    # condition labels
    @vlplot({:text, align = :left, dx = 5},
        transform = [{filter = "datum.winstart > 2.2"}],
        x = {datum = 3.0},
        y = {:logitmeandiff, aggregate = :mean, type = :quantitative},
        text = :chgroup
    ) +
    # Basline (0 %) dotted line
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:rule, strokeDash = [4 4], size = 2},
            y = {datum = 0.0},
            color = {value = "black"})
    ) +
    # "Null Model" text annotation
    # (
    #     @vlplot(data = {values = [{}]}) +
    #     @vlplot(mark = {:text, size = 11, baseline = "line-top", dy = 4},
    #         x = {datum = 2.3}, y = {datum = nullmean},
    #         text = {value = ["Mean Accuracy", "of Null Model"]},
    #         color = {value = "black"}
    #     )
    # ) +
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
        @vlplot(mark = {:text, size = 11, baseline = "bottom", yOffset = -3},
            x = {datum = 0.5}, y = {datum = target_len_y},
            text = {value = "Target Length"},
            color = {value = "black"}
        )
    ));
pl |> save(joinpath(dir, "salience_chgroup.svg"))

# Early/late targets
# =================================================================

# Mean Frequency Bin Analysis
# -----------------------------------------------------------------

classdf_earlylate_file = joinpath(cache_dir("features"), "salience-freqmeans-earlylate-timeline.csv")

if isfile(classdf_earlylate_file)
    classdf_earlylate = CSV.read(classdf_earlylate_file)
else
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    classdf_earlylate_groups = @_ events |>
        filter(ishit(_, region = "target") ∈ ["hit"], __) |>
        groupby(__, [:sid, :condition, :salience_label, :target_time_label])

    windows = [windowtarget(len = len, start = start)
        for len in 2.0 .^ range(-1, 1, length = 10),
            start in range(0, 2.5, length = 12)]
    classdf_earlylate = compute_freqbins(subjects, classdf_earlylate_groups, windows)

    CSV.write(classdf_earlylate_file, classdf_earlylate)
end

# Compute classification accuracy
# -----------------------------------------------------------------

resultdf_earlylate_file = joinpath(cache_dir("models"), "salience-earlylate-timeline.csv")
classdf_earlylate[!,:fold] = getindex.(Ref(fold_map), classdf_earlylate.sid)

if isfile(resultdf_earlylate_file) && mtime(resultdf_earlylate_file) > mtime(classdf_earlylate_file)
    resultdf_earlylate = CSV.read(resultdf_earlylate_file)
else
    factors = [:fold, :winlen, :winstart, :condition, :target_time_label]
    groups = groupby(classdf_earlylate, factors)

    progress = Progress(length(groups))
    function findclass((key, sdf))
        λ = λ_map[first(sdf.fold)]
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
    groupby(__, [:sid, :λ, :fold, :condition, :target_time_label]) |>
    combine(__, :mean => mean => :mean)

nullmeans = @_ classmeans_sum |>
    filter(_.λ == 1.0, __) |>
    rename!(__, :mean => :nullmean) |>
    deletecols!(__, :λ)

nullmean, classdiffs =
    let l = logit ∘ shrinktowards(0.5, by = 0.01),
        C = mean(l.(nullmeans.nullmean)),
        tocor = x -> logistic(x + C)
    logistic(C),
    @_ classmeans_sum |>
        filter(_.λ != 1.0, __) |>
        innerjoin(__, nullmeans, on = [:condition, :sid, :fold, :target_time_label]) |>
        transform!(__, [:mean, :nullmean] => ByRow((x,y) -> (l(x)-l(y))) => :logitmeandiff) |>
        groupby(__, [:condition, :target_time_label]) |>
        combine(__,
            :logitmeandiff => tocor ∘ mean => :mean,
            :logitmeandiff => (x -> tocor(lowerboot(x, alpha = 0.318))) => :lower,
            :logitmeandiff => (x -> tocor(upperboot(x, alpha = 0.318))) => :upper,
        ) |>
        transform!(__, [:condition, :target_time_label] => ByRow(string) => :condition_time)
end

ytitle = ["Neural Salience-Classification", "Accuracy (Null Model Corrected)"]
barwidth = 14
yrange = [0, 1]
pl = classdiffs |>
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
        color = {:condition_time, title = nothing, scale = {range = "#".*hex.(lightdark)}},
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
        @vlplot({:text, angle = 0, fontSize = 9, align = "left", baseline = "line-top",
            dx = -2barwidth - 17, dy = 22},
            color = {value = "#"*hex(darkgray)},
            x = {datum = "global"},
            y = {datum = yrange[1]},
            text = {datum = ["Less Distinct", "Response to Salience"]}
        ) +
        @vlplot({:text, angle = 0, fontSize = 9, align = "left", baseline = "line-bottom",
            dx = -2barwidth - 17, dy = -24},
            color = {value = "#"*hex(darkgray)},
            x = {datum = "global"},
            y = {datum = yrange[2]},
            text = {datum = ["More Distict", "Response to Salience"]}
        )
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
pl |> save(joinpath(dir, "salience_earlylate.svg"))

# Behavioral early/late salience
# -----------------------------------------------------------------

means = @_ CSV.read(joinpath(processed_datadir("plots"),
    "hitrate_angle_byswitch_andtarget.csv")) |>
    transform!(__, [:condition, :target_time] => ByRow(string) => :condition_time)

barwidth = 8
yrange = [0, 1]
pl = means |>
    @vlplot(
        # width = 121,
        spacing = 5,
        transform = [{filter = "datum.variable == 'hitrate'"}],
        config = {
            legend = {disable = true},
            bar = {discreteBandSize = barwidth}
        },
        facet = {
            column = {field = :salience, type = :nominal, title = nothing,
                sort = ["low", "high"],
                header = {labelFontWeight = "bold",
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1) + ' Salience'"}, },
        }
    ) + (
        @vlplot(width = 95, height = 150) +
        @vlplot({:bar, xOffset = -(barwidth/2), clip = true},
            transform = [{filter = "datum.target_time == 'early'"}],
            x = {:condition, axis = {title = "", labelAngle = -32,
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}, },
            y = {:pmean, title = "Hit Rate", scale = {domain = yrange}},
            color = {:condition_time, scale = {range = "#".*hex.(lightdark)}}
        ) +
        @vlplot({:rule, xOffset = -(barwidth/2)},
            transform = [{filter = "datum.target_time == 'early'"}],
            x = :condition,
            y = {:lowerc, title = ""}, y2 = :upperc,
            color = {value = "black"}
        ) +
        @vlplot({:bar, xOffset = (barwidth/2), clip = true},
            transform = [{filter = "datum.target_time == 'late'"}],
            x = :condition,
            y = {:pmean, title = ""},
            color = {:condition_time, scale = {range = "#".*hex.(lightdark)}}
        ) +
        @vlplot({:rule, xOffset = (barwidth/2)},
            transform = [{filter = "datum.target_time == 'late'"}],
            x = :condition,
            y = {:lowerc, title = ""}, y2 = :upperc,
            color = {value = "black"}
        ) +
        @vlplot({:text, angle = -90, fontSize = 9, align = "left", baseline = "bottom", dx = 0, dy = -barwidth-2},
            transform = [{filter = "datum.target_time == 'early' && datum.condition == 'global'"}],
            # x = {datum = "spatial"}, y = {datum = 0.},
            x = {:condition, axis = {title = ""}},
            y = {datum = yrange[1]},
            text = {value = "Early"},
        ) +
        @vlplot({:text, angle = -90, fontSize = 9, align = "right", baseline = "top", dx = 0, dy = barwidth+2},
            transform = [{filter = "datum.target_time == 'late' && datum.condition == 'global'"}],
            # x = {datum = "spatial"}, y = {datum = },
            x = {:condition, axis = {title = ""}},
            y = {:pmean, aggregate = :mean, type = :quantitative},
            text = {value = "Late"},
        )
    );
pl |> save(joinpath(dir, "behavior_earlylate_hitrate.svg"))

barwidth = 8
yrange = [-1, 40]
pl = means |>
    @vlplot(
        # width = 121,
        spacing = 5,
        transform = [{filter = "datum.variable == 'angle'"}],
        config = {
            legend = {disable = true},
            bar = {discreteBandSize = barwidth}
        },
        facet = {
            column = {field = :salience, type = :nominal, title = nothing,
                sort = ["low", "high"],
                header = {labelFontWeight = "bold",
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1) + ' Salience'"}, },
        }
    ) + (
        @vlplot(width = 98, height = 150) +
        @vlplot({:bar, xOffset = -(barwidth/2), clip = true},
            transform = [{filter = "datum.target_time == 'early'"}],
            x = {:condition, axis = {title = "", labelAngle = -32,
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}, },
            y = {:pmean, title = "Hit Rate Angle", scale = {domain = yrange}},
            color = {:condition_time, scale = {range = "#".*hex.(lightdark)}}
        ) +
        @vlplot({:rule, xOffset = -(barwidth/2)},
            transform = [{filter = "datum.target_time == 'early'"}],
            x = :condition,
            y = {:lowerc, title = ""}, y2 = :upperc,
            color = {value = "black"}
        ) +
        @vlplot({:bar, xOffset = (barwidth/2), clip = true},
            transform = [{filter = "datum.target_time == 'late'"}],
            x = :condition,
            y = {:pmean, title = ""},
            color = {:condition_time, scale = {range = "#".*hex.(lightdark)}}
        ) +
        @vlplot({:rule, xOffset = (barwidth/2)},
            transform = [{filter = "datum.target_time == 'late'"}],
            x = :condition,
            y = {:lowerc, title = ""}, y2 = :upperc,
            color = {value = "black"}
        ) +
        @vlplot({:text, angle = -90, fontSize = 9, align = "left", baseline = "bottom", dx = 0, dy = -barwidth-2},
            transform = [{filter = "datum.target_time == 'early' && datum.condition == 'global'"}],
            # x = {datum = "spatial"}, y = {datum = 0.},
            x = {:condition, axis = {title = ""}},
            y = {datum = 0},
            text = {value = "Early"},
        ) +
        @vlplot({:text, angle = -90, fontSize = 9, align = "left", baseline = "top", dx = 0, dy = barwidth+2},
            transform = [{filter = "datum.target_time == 'late' && datum.condition == 'global'"}],
            # x = {datum = "spatial"}, y = {datum = },
            x = {:condition, axis = {title = ""}},
            y = {datum = 0},
            text = {value = "Late"},
        )
    );
pl |> save(joinpath(dir, "behavior_earlylate_angle.svg"))

# Combine  behavioral and neural early/late responses
# -----------------------------------------------------------------

GermanTrack.@usepython

svg = pyimport("svgutils").compose

background_file = tempname()

background = pyimport("svgutils").transform.fromstring("""
    <svg>
        <rect width="100%" height="100%" fill="white"/>
    </svg>
""").save(background_file)

# NOTE: we have to make all of the `clipN` definitions distinct
# across the three files we're combining
for (suffix, file) in [
    ("behavior_hitrate", "behavior_earlylate_hitrate.svg"),
    ("behavior_angle", "behavior_earlylate_angle.svg"),
    ("neural", "salience_earlylate.svg")]
    filereplace(joinpath(dir, file), r"\bclip([0-9]+)\b" =>
        SubstitutionString("clip\\1_$suffix"))
end

fig = svg.Figure("89mm", "235mm",
    svg.SVG(background_file),
    svg.Panel(
        svg.SVG(joinpath(dir, "behavior_earlylate_hitrate.svg")).move(0,15),
        svg.Text("A", 2, 10, size = 12, weight="bold"),
        svg.SVG(joinpath(plotsdir("icons"), "behavior.svg")).
            scale(0.1).move(115,35),
        svg.SVG(joinpath(plotsdir("icons"), "behavior.svg")).
            scale(0.1).move(220,35)
    ).move(0, 0),
    svg.Panel(
        svg.SVG(joinpath(dir, "behavior_earlylate_angle.svg")).move(0,15),
        svg.Text("B", 2, 10, size = 12, weight = "bold"),
        svg.SVG(joinpath(plotsdir("icons"), "behavior.svg")).
            scale(0.1).move(115,35),
        svg.SVG(joinpath(plotsdir("icons"), "behavior.svg")).
            scale(0.1).move(220,35)
    ).move(0, 235),
    svg.Panel(
        svg.SVG(joinpath(dir, "salience_earlylate.svg")).move(0,15),
        svg.Text("C", 2, 10, size = 12, weight = "bold"),
        svg.SVG(joinpath(plotsdir("icons"), "eeg.svg")).
            scale(0.1).move(220,40)
    ).move(0, 235 + 235),
).scale(1.333).save(joinpath(dir, "fig4.svg"))

# Plot 4-salience-level, trial-by-trial timeline
# =================================================================

# Compute frequency bins
# -----------------------------------------------------------------

classdf_sal4_timeline_file = joinpath(cache_dir("features"), "salience-4level-freqmeans-timeline-trial.csv")

if isfile(classdf_sal4_timeline_file)
    classdf_sal4_timeline = CSV.read(classdf_sal4_timeline_file)
else
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    classdf_sal4_timeline_groups = @_ events |>
        transform!(__, AsTable(:) => ByRow(x -> ishit(x, region = "target")) => :hittype) |>
        filter(_.hittype ∈ ["hit", "miss"], __) |>
        filter(_.condition == "global", __) |>
        groupby(__, [:sid, :condition, :salience_4level, :trial, :hittype])

    windows = [
        windowtarget(windowfn = event -> (
            start = start,
            len = winlen_bysid(event.sid[1]) |> GermanTrack.spread(0.5,n_winlens,indices=k)))
        for start in range(0, 3, length = 64) for k in 1:n_winlens
    ]
    classdf_sal4_timeline =
        compute_freqbins(subjects, classdf_sal4_timeline_groups, windows)

    CSV.write(classdf_sal4_timeline_file, classdf_sal4_timeline)
end

# Compute classification accuracy
# -----------------------------------------------------------------

# QUESTION: should we reselect lambda?

λsid = groupby(final_λs, :sid)

resultdf_sal4_timeline_file = joinpath(cache_dir("models"), "salience-timeline-sal4.csv")
classdf_sal4_timeline[!,:fold] = getindex.(Ref(fold_map), classdf_sal4_timeline.sid)

levels = ["lowest","low","high","highest"]
classdf_sal4_timeline[!,:salience_4label] =
    CategoricalArray(get.(Ref(levels),coalesce.(classdf_sal4_timeline.salience_4level,0),missing), levels = levels)

modeltype = [ "1v4" => [1,4], "1v3" => [1,3], "1v2" => [1,2] ]

if isfile(resultdf_sal4_timeline_file) && mtime(resultdf_sal4_timeline_file) > mtime(classdf_sal4_timeline_file)
    resultdf_sal4_timeline = CSV.read(resultdf_sal4_timeline_file)
else
    factors = [:fold, :winlen, :winstart, :condition, :hittype]
    groups = groupby(classdf_sal4_timeline, factors)

    progress = Progress(length(groups) * length(modeltype))

    function findclass(((key, sdf), (type, levels)))
        λ = λ_map[first(sdf.fold)]
        filtered = @_ sdf |> filter(_.salience_4level ∈ levels, __)

        result = testclassifier(LassoPathClassifiers([1.0, λ]),
            data = filtered, y = :salience_4label, X = r"channel", crossval = :sid,
            n_folds = n_folds, seed = stablehash(:salience_classification, 2019_11_18),
            maxncoef = size(filtered[:,r"channel"], 2),
            ycoding = StatsModels.SeqDiffCoding,
            irls_maxiter = 600, weight = :weight, on_model_exception = :throw)

        result[!, keys(key)]  .= permutedims(collect(values(key)))
        result[!, :modeltype] .= type
        next!(progress)

        result
    end

    resultdf_sal4_timeline = @_ groups |> pairs |> collect |>
        Iterators.product(__, modeltype) |>
        # foldl(append!!, Map(findclass), __)
        foldxt(append!!, Map(findclass), __)

    ProgressMeter.finish!(progress)
    CSV.write(resultdf_sal4_timeline_file, resultdf_sal4_timeline)

    alert("Completed salience timeline classification!")
end

# Display classification timeline
# -----------------------------------------------------------------

classmeans = @_ resultdf_sal4_timeline |>
    groupby(__, [:winstart, :winlen, :sid, :λ, :fold, :condition, :modeltype, :hittype]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :mean)

classmeans_sum = @_ classmeans |>
    groupby(__, [:winstart, :sid, :λ, :fold, :condition, :modeltype, :hittype]) |>
    combine(__, :mean => mean => :mean)

nullmeans = @_ classmeans_sum |>
    filter(_.λ == 1.0, __) |>
    rename!(__, :mean => :nullmean) |>
    deletecols!(__, :λ)

sallevel_pairs = [
    "1v4" => "High:\t1st vs. 4th Quartile",
    "1v3" => "Medium:\t1st vs. 3rd Quartile",
    "1v2" => "Low:\t1st vs. 2nd Quartile",
]
sallevel_shortpairs = [
    "1v4" => ["High" , "1st vs. 4th Q"],
    "1v3" => ["Medium", "1st vs. 3rd Q"],
    "1v2" => ["Low", "1st vs. 2nd Q"],
]

nullmean, classdiffs = let l = logit ∘ shrinktowards(0.5, by = 0.01), C = mean(l.(nullmeans.nullmean))
    100logistic(C),
    @_ classmeans_sum |>
        filter(_.λ != 1.0, __) |>
        innerjoin(__, nullmeans, on = [:winstart, :condition, :sid, :fold, :modeltype, :hittype]) |>
        transform!(__, [:mean, :nullmean] => ByRow((x,y) -> l(x) - l(y)) => :logitmeandiff) |>
        transform!(__, [:mean, :nullmean] => ByRow((x,y) -> 100logistic(l(x) - l(y) + C)) => :meancor) |>
        transform!(__, :condition => ByRow(uppercasefirst) => :condition) |>
        transform!(__, :modeltype => (x -> replace(x, sallevel_pairs...)) => :modeltype_title) |>
        transform!(__, :modeltype => (x -> replace(x , sallevel_shortpairs...)) => :modeltype_shorttitle)
end

annotate = @_ map(abs(_ - 3.0), classdiffs.winstart) |> classdiffs.winstart[argmin(__)]
ytitle = ["% Correct", "(Null-Model Corrected)"]
target_length_y = 70.0
pl = classdiffs |>
    @vlplot(
        title = {text = "Salience Classification Accuracy", subtitle = "Global Condition"},
        config = {
            legend = {orient = :none, legendX = 5, legendY = 0.5, title = "Classification"},
        },
        facet = {column = {field = :hittype, type = :nominal,
                 title = nothing}}
    ) +
    (
        @vlplot(color = {field = :modeltype_title, type = :nominal,
            scale = {scheme = :inferno}, sort = getindex.(sallevel_pairs, 2)}) +
        # data lines
        @vlplot(:line,
            x = {:winstart, type = :quantitative, title = "Time (s) Relative to Target Onset"},
            y = {:meancor, aggregate = :mean, type = :quantitative, title = ytitle,
                 scale = {domain = [50, 100]}
                }) +
        # data errorbands
        @vlplot(:errorband,
            x = {:winstart, type = :quantitative, title = "Time (s) Relative to Target Onset"},
            y = {:meancor, aggregate = :ci, type = :quantitative, title = ytitle}) +
        # condition labels
        @vlplot({:text, align = :left, dx = 5},
            transform = [{filter = "datum.winstart > 2.4 && datum.winstart < 2.5"}],
            x = {datum = 3.0},
            y = {:meancor, aggregate = :mean, type = :quantitative},
            text = :modeltype_shorttitle
        ) +
        # Basline (0 %) dotted line
        (
            @vlplot(data = {values = [{}]}) +
            @vlplot(mark = {:rule, strokeDash = [4 4], size = 2},
                y = {datum = nullmean},
                color = {value = "black"})
        ) +
        # "Target Length" arrow annotation
        (
            @vlplot(data = {values = [
                {x = 0.05, y = target_length_y, dir = 270},
                {x = 0.95, y = target_length_y, dir = 90}]}) +
            @vlplot(mark = {:line, size = 1.5},
                x = {:x, type = :quantitative},
                y = {:y, type = :quantitative},
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
                x = {datum = 0.5}, y = {datum = target_length_y},
                text = {value = "Target Length"},
                color = {value = "black"}
            )
        ) +
        # "Null Model" text annotation
        (
            @vlplot(data = {values = [{}]}) +
            @vlplot(mark = {:text, size = 11, baseline = "line-top", dy = 4},
                x = {datum = 2.0}, y = {datum = nullmean},
                text = {value = ["Mean Accuracy", "of Null Model"]},
                color = {value = "black"}
            )
        )
    );
pl |> save(joinpath(dir, "salience_timeline_4level.svg"))

hitvmiss = @_ classdiffs |>
    unstack(__, [:winstart, :sid, :condition, :modeltype], :hittype, :logitmeandiff) |>
    transform!(__, [:hit, :miss] => (-) => :hitvmiss) |>
    transform!(__, :modeltype => (x -> replace(x, sallevel_pairs...)) => :modeltype_title) |>
    transform!(__, :modeltype => (x -> replace(x , sallevel_shortpairs...)) => :modeltype_shorttitle)

annotate = @_ map(abs(_ - 3.0), classdiffs.winstart) |> classdiffs.winstart[argmin(__)]
ytitle = "Hit Accuracy - Miss Accuracy"
pl = hitvmiss |>
    @vlplot(
        # config = {legend = {disable = true}},
        config = {
            legend = {orient = :none, legendX = 5, legendY = 0.5, title = "Classification"},
        },
        title = ["Low/High Salience Classification Accuracy", "Object Condition"]
        # facet = {column = {field = :hittype, type = :nominal}}
    ) +
    (@vlplot(
        color = {field = :modeltype_title, type = :nominal, scale = {scheme = :inferno}},
    ) +
    # data lines
    @vlplot(:line,
        x = {:winstart, type = :quantitative, title = "Time (s) Relative to Target Onset"},
        y = {:hitvmiss, aggregate = :mean, type = :quantitative, title = ytitle}) +
    # data errorbands
    @vlplot(:errorband,
        x = {:winstart, type = :quantitative, title = "Time (s) Relative to Target Onset"},
        y = {:hitvmiss, aggregate = :ci, type = :quantitative, title = ytitle}) +
    # condition labels
    @vlplot({:text, align = :left, dx = 5},
        transform = [{filter = "datum.winstart > 2.05 && datum.winstart < 2.11"}],
        x = {datum = 3.0},
        y = {:hitvmiss, aggregate = :mean, type = :quantitative},
        text = :modeltype_shorttitle
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
            {x = 0.05, y = 0.25, dir = 270},
            {x = 0.95, y = 0.25, dir = 90}]}) +
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
            x = {datum = 0.5}, y = {datum = 0.25},
            text = {value = "Target Length"},
            color = {value = "black"}
        )
    ));
pl |> save(joinpath(dir, "salience_timeline_4level_hitvmiss.svg"))

