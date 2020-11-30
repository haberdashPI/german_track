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
        smoothing = 0.85, slope_thresh = 0.15, flat_thresh = 0.01,
        dir = mkpath(joinpath(dir, "supplement")))

    @info "Saving plots to $(joinpath(dir, "supplement"))"
end

# Plot timeline (Fig 3d)
# =================================================================

# Compute classificaiton accuracy
# -----------------------------------------------------------------

file = joinpath(cache_dir("features"), "salience-freqmeans-timeline.json")
GermanTrack.@cache_results file resultdf_timeline begin
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
        combine(__, [:correct, :weight] => GermanTrack.wmean => :mean,
                    :weight => sum => :weight,
                    :correct => length => :count) |>
        transform(__, :weight => (x -> x ./ mean(x)) => :weight)

classmeans_sum = @_ classmeans |>
    groupby(__, [:winstart, :sid, :λ, :fold, :condition, :hittype]) |>
    combine(__,
        [:mean, :weight] => GermanTrack.wmean => :mean,
        :weight => sum => :weight,
        :count => length => :count)

nullmeans = @_ classmeans_sum |>
    filter(_.λ == 1.0, __) |>
    rename!(__, :mean => :nullmean) |>
    deletecols!(__, [:λ, :weight, :count])

statdata = @_ classmeans_sum |>
    filter(_.λ != 1.0, __) |>
    innerjoin(__, nullmeans, on = [:winstart, :condition, :sid, :fold, :hittype]) |>
    transform!(__, :nullmean => ByRow(logit ∘ shrinktowards(0.5, by = 0.01)) => :logitnullmean)

CSV.write(processed_datadir("analyses", "eeg_salience_timeline.csv"), statdata)

file = processed_datadir("analyses", "eeg_salience_timeline_correct.csv")
corrected_data = @_ CSV.read(file, DataFrame) |>
    @transform(__, condition_label = uppercasefirst.(:condition))
nullmean = 0.53

timeslice = @_ corrected_data |> groupby(__, [:winstart, :condition]) |>
    @combine(__, mean = mean(:corrected_mean)) |>
    groupby(__, :winstart) |>
    @combine(__, mean = maximum(:mean)) |>
    @combine(__, best = :winstart[argmax(:mean)]) |>
    __.best |> first

ytitle = ["High/Low Salience Classification", "Accuracy (Null Model Corrected)"]
target_len_y = 0.8
pl = @_ corrected_data |>
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
        y = {:corrected_mean, aggregate = :mean, type = :quantitative, title = ytitle,
            scale = {domain = [0.5,1.0]}}) +
    # data errorbands
    @vlplot(:errorband,
        x = {:winstart, type = :quantitative},
        y = {:corrected_mean, aggregate = :ci, type = :quantitative, title = ytitle}) +
    # condition labels
    @vlplot({:text, align = :left, dx = 5},
        transform =
            [{filter = "(datum.condition != 'spatial' && "*
                "datum.winstart > 2.95 && datum.winstart <= 3.0) ||"*
                "(datum.condition == 'spatial' && "*
                "datum.winstart > 2.45 && datum.winstart <= 2.55)"}],
        x = {datum = 3.0},
        y = {:corrected_mean, aggregate = :mean, type = :quantitative},
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

# Salience class accuracy at fixed time point (fig 3c)
# =================================================================

times = corrected_data.winstart |> unique
real_timeslice = times[argmin(abs.(times .- timeslice))]
barwidth = 16

ytitle = ["High/Low Salience Classification", "Accuracy (Null Model Corrected)"]
yrange = [0.5, 1]
pl = @_ corrected_data |>
    filter(_.hittype == "hit", __) |>
    filter(_.winstart == real_timeslice, __) |>
    groupby(__, [:condition]) |>
    combine(__,
        :corrected_mean => mean => :corrected_mean,
        :corrected_mean => lowerboot => :lower,
        :corrected_mean => upperboot => :upper,
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
        y = {:corrected_mean,
            title = ytitle,
            scale = {domain = yrange}
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
    );
plotfile = joinpath(dir, "fig3c.svg")
pl |> save(plotfile)
addpatterns(plotfile, seqpatterns, size = 10)

# Final, combined plots for data fig 2
# =================================================================

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
).scale(1.333).save(joinpath(plotsdir("figures"), "fig3.svg"))
