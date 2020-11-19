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

