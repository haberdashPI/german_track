# Setup
# =================================================================

n_winlens = 12
n_folds = 10

using DrWatson
@quickactivate("german_track")
using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures, Dates, Underscores,
    Printf, ProgressMeter, VegaLite, FileIO, StatsBase, BangBang, Transducers,
    Infiltrator, Peaks, Distributions, DSP, Random, CategoricalArrays, StatsModels,
    StatsFuns, Colors, CSV, DataFramesMeta

using GermanTrack: colors, gray, patterns, lightdark, darkgray, inpatterns, allcols

dir = mkpath(plotsdir("figure5_parts"))
dir_supplement = mkpath(plotsdir("figure5_parts","supplement"))

# Behavioral early/late salience (fig 5a)
# =================================================================

means = @_ CSV.read(joinpath(processed_datadir("plots"),
    "hitrate_angle_byswitch_andtarget.csv"), DataFrame) |>
    transform!(__, [:condition, :target_time, :salience] => ByRow(string) => :condition_time_salience)
cts_order = [
     "globalearlylow", "globalearlyhigh" , "globallatelow" , "globallatehigh" ,
     "objectearlylow", "objectearlyhigh" , "objectlatelow" , "objectlatehigh" ,
    "spatialearlylow", "spatialearlyhigh", "spatiallatelow", "spatiallatehigh",
]

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
            color = {
                type = :ordinal,
                :condition_time_salience, scale = {
                    sort = cts_order,
                    range = "#".*hex.(allcols)
                },
            }
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
            color = {:condition_time_salience, scale = {
                sort = cts_order,
                range = "#".*hex.(allcols)
            },
}
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
pl |> save(joinpath(dir, "fig5a.svg"))

# TODO: Figure 5b: difference of hit rate angle

# TODO: code below is out of date, and will not run
# I'm not revising it because we're changing how this part is being analyzed
# Early/late salience classification (Figure 5c)
# =================================================================

# Compute classificaiton accuracy
# -----------------------------------------------------------------

file = joinpath(processed_datadir("analyses"), "salience-hyperparams.json")
GermanTrack.@load_cache file fold_map hyperparams

file = joinpath(processed_datadir("analyses"), "salience-earlylate.json")
GermanTrack.@cache_results file resultdf_earlylate begin
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    # TODO: load fold-selected starting window
    lens = @_ getindex.(values(hyperparams), :winlen) |> unique |>
        GermanTrack.spread.(__, 0.5, n_winlens) |> reduce(vcat, __) |> unique
    winstart = 3.4

    classdf = @_ events |>
        transform!(__, AsTable(:) => ByRow(findresponse) => :hittype) |>
        transform!(__, :sid => ByRow(x -> fold_map[x]) => :fold) |>
        filter(_.hittype ∈ ["hit"], __) |>
        groupby(__, [:sid, :fold, :condition, :salience_label, :target_time_label, :hittype]) |>
        filteringmap(__, desc = "Computing features...",
            :window => [
                windowtarget(start = winstart, len = len)
                for len in lens
            ],
            compute_powerbin_features(_1, subjects, _2)) |>
        delete!(__, :window)

    resultdf_earlylate = @_ classdf |>
        groupby(__, [:condition, :target_time_label]) |>
        filteringmap(__,
            folder = foldl,
            desc = "Classifying early/late salience...",
            :cross_fold => 1:10,
            :modeltype => ["full", "null"],
            function (sdf, fold, modeltype)
                selector = modeltype == "null" ? m -> NullSelect() : hyperparams[fold][:λ]
                lens = hyperparams[fold][:winlen] |> GermanTrack.spread(0.5, n_winlens)
                # TODO: will need to also filter out proper winstarts where

                sdflens = filter(x -> x.winlen ∈ lens, sdf)
                result = combine(groupby(sdflens, :winlen)) do sdf_len
                    test, model = traintest(sdf_len, fold, y = :salience_label,
                        selector = selector, weight = :weight)
                    test[:, Not(r"channel")]
                end

                result
            end)
end

# Plot salience by early/late targets
# -----------------------------------------------------------------

classmeans = @_ resultdf_earlylate |>
    groupby(__, [:winstart, :winlen, :sid, :modeltype, :fold, :condition, :target_time_label]) |>
    @combine(__,
        mean   = GermanTrack.wmean(:correct, :weight),
        weight = sum(:weight),
        count  = length(:correct)
    )

classmeans_sum = @_ classmeans |>
    groupby(__, [:sid, :modeltype, :fold, :condition, :target_time_label]) |>
    @combine(__,
        mean = mean(:mean),
        weight = sum(:weight),
        count = length(:mean)
    )

nullmeans = @_ classmeans_sum |>
    filter(_.modeltype == "null", __) |>
    rename!(__, :mean => :nullmean) |>
    delete!(__, [:modeltype, :weight, :count])

statdata = @_ classmeans_sum |>
    filter(_.modeltype == "full", __) |>
    innerjoin(__, nullmeans, on = [:condition, :sid, :fold, :target_time_label]) |>
    @transform(__, logitnullmean = logit.(shrinktowards.(:nullmean, 0.5, by = 0.01))) |>
    @transform(__, logitmean = logit.(shrinktowards.(:mean, 0.5, by = 0.01)))
CSV.write(processed_datadir("analyses", "eeg_salience_earlylate.csv"), statdata)

nullmean = logistic.(mean(logit.(shrinktowards.(nullmeans.nullmean, 0.5, by = 0.01))))

# supplemental figure

pl = statdata |>
    @vlplot(:point,
        column = :condition,
        color = :target_time_label,
        x = :logitnullmean,
        y = :logitmean
    );
pl |> save(joinpath(dir_supplement, "earlylate_ind.svg"))

run(`Rscript $(joinpath(scriptsdir("R"), "salience_earlylate.R"))`)

file = processed_datadir("analyses", "eeg_salience_earlylate_coefs.csv")
classdiffs = @_ CSV.read(file, DataFrame) |>
    rename(__, :r_med => :mean, :r_05 => :lower, :r_95 => :upper) |>
    @transform(__,
        condition_time = :comparison,
        condition = getindex.(split.(:comparison, "_"),1),
        target_time_label = getindex.(split.(:comparison, "_"),2)
    )

ytitle = ["High/Low Salience", "Classification"]
barwidth = 14
yrange = [0.1, 1]
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
plotfile = joinpath(dir, "fig5c.svg")
pl |> save(plotfile)
addpatterns(plotfile, inpatterns, size = 10)

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
    ("behavior_hitrate", "fig5a.svg"),
    # ("behavior_angle", "behavior_earlylate_angle.svg"),
    ("neural", "fig5c.svg")]
    filereplace(joinpath(dir, file), r"\bclip([0-9]+)\b" =>
        SubstitutionString("clip\\1_$suffix"))
end

fig = svg.Figure("89mm", "235mm",
    svg.SVG(background_file),
    svg.Panel(
        svg.SVG(joinpath(dir, "fig5a.svg")).move(0,15),
        svg.Text("A", 2, 10, size = 12, weight="bold"),
        svg.SVG(joinpath(plotsdir("icons"), "behavior.svg")).
            scale(0.1).move(115,35),
        svg.SVG(joinpath(plotsdir("icons"), "behavior.svg")).
            scale(0.1).move(220,35)
    ).move(0, 0),
    # svg.Panel(
    #     svg.SVG(joinpath(dir, "behavior_earlylate_angle.svg")).move(0,15),
    #     svg.Text("B", 2, 10, size = 12, weight = "bold"),
    #     svg.SVG(joinpath(plotsdir("icons"), "behavior.svg")).
    #         scale(0.1).move(115,35),
    #     svg.SVG(joinpath(plotsdir("icons"), "behavior.svg")).
    #         scale(0.1).move(220,35)
    # ).move(0, 235),
    svg.Panel(
        svg.SVG(joinpath(dir, "fig5c.svg")).move(0,15),
        svg.Text("C", 2, 10, size = 12, weight = "bold"),
        svg.SVG(joinpath(plotsdir("icons"), "eeg.svg")).
            scale(0.1).move(220,15)
    ).move(0, 235 + 235),
).scale(1.333).save(joinpath(plotsdir("figures"), "fig5.svg"))

