# Setup
# =================================================================

n_winlens = 12
n_folds = 10

using DrWatson; @quickactivate("german_track")
using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures, Dates, Underscores,
    Printf, ProgressMeter, VegaLite, FileIO, StatsBase, BangBang, Transducers,
    Infiltrator, Peaks, Distributions, DSP, Random, CategoricalArrays, StatsModels,
    StatsFuns, Colors, CSV, DataFramesMeta, ColorSchemes

# using GermanTrack: colors, gray, patterns, lightdark, darkgray, seqpatterns, neutral

points = [0.25,mean([0.25,0.7]),0.7,0.85]
jobcolors = ColorSchemes.batlow[points]
patpoints = Iterators.flatten([points[i], points[i] + (points[i+1]-points[i])/3] for i in 1:3) |> collect
jobpattern_cols = ColorSchemes.batlow[patpoints]

jobpatterns = OrderedDict(
    "stripe1" => jobpattern_cols[[1,2]],
    "stripe2" => jobpattern_cols[[3,4]],
    "stripe3" => jobpattern_cols[[5,6]],
)

dir = mkpath(plotsdir("figure3_job"))

# Hit rate x salience level (Figure 3A)
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

barwidth = 10
pl = @_ diffmeans |>
    @vlplot(
        width = 79, autosize = "fit", height = 135,
        config = {
            legend = {disable = true},
            bar = {discreteBandSize = barwidth}
        }
    ) +
    @vlplot(:bar,
        x = {:condition,
            title = "",
            axis = {labelAngle = -45, labelAlign = "right",
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}
        },
        color = {:condition, scale = {
            range = urlcol.(keys(jobpatterns))}},
        y = {:meandiff,
            title = "Hit Rate (High - Low Sal.)",
            axis = {titleFontWeight = "normal"}
        }
    ) +
    @vlplot(:errorbar,
        x = {:condition, title = ""},
        y = {:lower, title = ""}, y2 = :upper,
    );
plotfile =  joinpath(dir, "fig3b.svg")
pl |> save(plotfile)
addpatterns(plotfile, jobpatterns, size = 10)

# EEG results
# =================================================================

file = joinpath(processed_datadir("analyses"), "salience-freqmeans-timeline.json")
if !@isdefined(resultdf_timeline)
    GermanTrack.@load_cache file resultdf_timeline
end

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
    @transform(__, logitnullmean = logit.(shrinktowards.(:nullmean, 0.5, by = 0.01)))

# CSV.write(processed_datadir("analyses", "eeg_salience_timeline.csv"), statdata)

logitnullmean = mean(statdata.logitnullmean)
nullmean = logistic.(logitnullmean)
corrected_data = @_ statdata |>
    @transform(__,
        corrected_mean =
            logistic.(logit.(shrinktowards.(:mean, 0.5, by = 0.01)) .-
                :logitnullmean .+ logitnullmean),
        condition_label = uppercasefirst.(:condition))

maxdiff(xs) = maximum(abs(xs[i] - xs[j]) for i in 1:(length(xs)-1) for j in (i+1):length(xs))
timeslice = @_ corrected_data |> groupby(__, [:winstart, :condition, :fold]) |>
    @combine(__, mean = mean(:corrected_mean)) |>
    groupby(__, [:winstart, :condition]) |>
    filteringmap(__,
        :train_fold => map(fold -> fold => (sdf -> sdf.fold != fold),
            unique(corrected_data.fold)),
        (sdf, fold) -> DataFrame(foldmean = mean(sdf.mean))) |>
    groupby(__, [:winstart, :train_fold]) |>
    @combine(__, score = maximum(:foldmean)) |>
    groupby(__, :train_fold) |>
    @combine(__, best = :winstart[argmax(:score)])
labelfn(fold) = "fold $fold"

ytitle = "Salience Decoding"
target_len_y = 0.83
pl = @_ corrected_data |>
    filter(_.hittype == "hit", __) |>
    @vlplot(
        width = 161, height = 175, autosize = "fit",
        config = {legend = {disable = true}},
    ) +
    (@vlplot(
        color = {field = :condition, type = :nominal, scale = {range = "#".*hex.(jobcolors)}},
    ) +
    # data lines
    @vlplot({:line, strokeCap = :round},
        strokeDash = {:condition, type = :nominal, scale = {range = [[1, 0], [6, 4], [2, 4]]}},
        x = {:winstart, type = :quantitative, title = "Time from onset (s)",
            axis = {titleFontWeight = "normal"}},
        y = {:corrected_mean, aggregate = :mean, type = :quantitative, title = ytitle,
            scale = {domain = [0.5,1.0]}, axis = {titleFontWeight = "normal"}}) +
    # data errorbands
    @vlplot(:errorband,
        x = {:winstart, type = :quantitative},
        y = {:corrected_mean, aggregate = :ci, type = :quantitative, title = ""}) +
    # condition labels
    @vlplot({:text, align = :left, dx = 5},
        transform =
            [{filter = "(datum.condition != 'object' && "*
                "datum.winstart > 2.95 && datum.winstart <= 3.0) ||"*
                "(datum.condition == 'object' && "*
                "datum.winstart > 2.85 && datum.winstart <= 2.95)"}],
        x = {datum = 3.0},
        y = {:corrected_mean, aggregate = :mean, type = :quantitative},
        text = :condition_label
    ) +
    # "Time Slice" annotation
    (
        @transform(timeslice, fold_label = labelfn.(:train_fold)) |>
        @vlplot() +
        @vlplot({:rule, strokeDash = [2 2]},
            x = "mean(best)",
            color = {value = "black"}
        ) +
        @vlplot({:text, align = "left", fontSize = 9, baseline = "bottom", angle = 90},
            x = "mean(best)",
            y = {datum = 1.0},
            text = {value = "Panel C"},
            color = {value = "#"*hex(neutral)}
        )
    ) +
    # (
    #     @vlplot(data = {values = [{}]}) +
    #     @vlplot({:text, align = "right", dx = -2},
    #         x = {datum = minimum(timeslice.best)},
    #         y = {datum = 0.95},
    #         text = {value = "Panel C slice →"},
    #         color = {value = "black"}
    #     )
    # ) +
    # # "Null Model" text annotation
    # (
    #     @vlplot(data = {values = [{}]}) +
    #     # white rectangle to give text a background
    #     @vlplot(mark = {:text, size = 11, baseline = "middle", dy = -5, dx = 5,
    #         align = "left"},
    #         x = {datum = 3}, y = {datum = nullmean},
    #         text = {value = ["Null Model", "Accuracy"]},
    #         color = {value = "black"}
    #     )
    # ) +
    # Dotted line
    # (
    #     @vlplot(data = {values = [{}]}) +
    #     @vlplot(mark = {:rule, strokeDash = [4 4], size = 2},
    #         y = {datum = nullmean},
    #         color = {value = "black"})
    # ) +
    # "Target Length" arrow annotation
    (
        @vlplot(data = {values = [
            {x = 0.1, y = target_len_y, dir = 270},
            {x = 0.9, y = target_len_y, dir = 90}]}) +
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

timeslice_map = Dict(row.train_fold => row.best for row in eachrow(timeslice))

barwidth = 10

ytitle = ["Salience Decoding"]
yrange = [0.5, 1]
pl = @_ corrected_data |>
    filter(_.hittype == "hit", __) |>
    filter(_.winstart == timeslice_map[_.fold], __) |>
    groupby(__, [:condition]) |>
    @combine(__,
        corrected_mean = mean(:corrected_mean),
        lower = lowerboot(:corrected_mean, alpha = 0.318),
        upper = upperboot(:corrected_mean, alpha = 0.318),
    ) |>
    @vlplot(
        width = 75, autosize = "fit", height = 135,
        config = {
            legend = {disable = true},
            bar = {discreteBandSize = barwidth}
        }
    ) +
    @vlplot(:bar,
        x = {:condition,
            title = "",
            axis = {labelAngle = -45, labelAlign = "right",
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}
        },
        color = {:condition, scale = {range = urlcol.(keys(jobpatterns))}},
        y = {:corrected_mean,
            title = ytitle,
            axis = {titleFontWeight = "normal"},
            scale = {domain = yrange}
        }
    ) +
    @vlplot(:errorbar,
        x = {:condition, title = ""},
        y = {:lower, title = ""}, y2 = :upper,
    );
plotfile = joinpath(dir, "fig3c.svg")
pl |> save(plotfile)
addpatterns(plotfile, jobpatterns, size = 10)

# Final, combined plots for data fig 2
# =================================================================

GermanTrack.@usepython

svg = pyimport("svgutils").compose

background_file = tempname()

schematic_ratio = 0.12

background = pyimport("svgutils").transform.fromstring("""
    <svg>
        <rect width="100%" height="100%" fill="white"/>
    </svg>
""").save(background_file)

fig = svg.Figure("57mm", "95mm",
    svg.SVG(background_file),
    svg.Panel(
        svg.SVG(joinpath(plotsdir("figures","job"), "fig3a.svg")).scale(schematic_ratio).move(0,5),
        # svg.Text("A", 2, 10, size = 12, weight="bold"),
    ).move(0, 0),
    # svg.Panel(
    #     svg.SVG(joinpath(dir, "fig3b.svg")).move(0,20),
    #     svg.Text("B", 2, 10, size = 12, weight="bold"),
    #     # svg.SVG(joinpath(plotsdir("icons"), "behavior.svg")).
    #     #     scale(0.1).move(90,15)
    # ).move(0, 98),
    # svg.Panel(
    #     svg.SVG(joinpath(dir, "fig3c.svg")).move(0,20),
    #     svg.Text("C", 2, 10, size = 12, weight = "bold"),
    #     # svg.SVG(joinpath(plotsdir("icons"), "eeg.svg")).
    #     #     scale(0.1).move(90,15)
    # ).move(85, 98),
    svg.Panel(
        svg.SVG(joinpath(dir, "fig3d.svg")).move(0,5),
        # svg.Text("B", 2, 10, size = 12, weight = "bold"),
        svg.SVG(joinpath(plotsdir("icons"), "eeg.svg")).
            scale(0.1).move(215,15)
    ).move(0, 98)
).scale(1.25).save(joinpath(plotsdir("figures","job"), "fig3.svg"))
