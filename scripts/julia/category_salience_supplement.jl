
dir = mkpath(plotsdir("figure3_parts", "supplement"))

# Re-analyze from raw data
# -----------------------------------------------------------------

info = GermanTrack.load_behavioral_stimulus_metadata()
events = @_ readdir(processed_datadir("behavioral"), join=true) |>
    filter(occursin(r"csv$", _), __) |>
    mapreduce(GermanTrack.events(_, info), append!!, __)

bad_sids = CSV.read(joinpath(processed_datadir("behavioral", "outliers"), "sids.csv")).sid

indmeans = @_ events |>
    transform!(__, AsTable(:) =>
        ByRow(x -> ishit(x, region = "target", mark_false_targets = true)) => :hittype) |>
    groupby(__, [:condition, :sid, :salience_label]) |>
    combine(__, :hittype => (x -> mean(==("hit"), x)) => :hits)

means = @_ indmeans |>
    filter(_.sid ∉ bad_sids, __) |>
    groupby(__, [:condition, :salience_label]) |>
    combine(__, :hits => boot => :prop,
                :hits => (x -> lowerboot(x, alpha = 0.05)) => :lower,
                :hits => (x -> upperboot(x, alpha = 0.05)) => :upper)

@_ means |>
    @vlplot(
        config = {legend = {disable = true}},
        spacing = 2
    ) +
    hcat(
        (
            @vlplot(
                width = 90, height = 125, autosize = "pad",
                transform = [{filter = "datum.condition == 'global' || datum.condition == 'object'"}]) +
            @vlplot(:line,
                x = {:condition, axis = {labelAngle = 0, labelAlign = "center", title = ""}},
                y = {:prop, title = "Hit Rate", scale = {domain = [0, 1]}},
                color = :salience_label) +
            @vlplot({:point, filled = true},
                x = :condition,
                y = :prop,
                color = {:condition, scale = {range = "#".*hex.(vcat(colors[1], RGB(0.3,0.3,0.3), RGB(0.6,0.6,0.6), colors[2:3]))}}) +
            @vlplot({:text, align = :left, dy = -10, dx = 5},
                transform = [{filter = "datum.condition == 'global'"}],
                x = :condition,
                y = :lower,
                text = :salience_label,
                color = :salience_label
            ) +
            @vlplot(:rule, x = :condition, y = :lower, y2 = :upper, color = :condition)
        ),
        (
            @vlplot(
                width = 90, height = 125, autosize = "pad",
                transform = [{filter = "datum.condition == 'global' || datum.condition == 'spatial'"}]
            ) +
            @vlplot(:line,
                x = {:condition, axis = {labelAngle = 0, labelAlign = "center", title = ""}},
                y = {:prop, title = "", scale = {domain = [0, 1]}},
                color = :salience_label) +
            @vlplot({:point, filled = true},
                x = :condition,
                y = :prop,
                color = {:condition, scale = {range = "#".*hex.(vcat(RGB(0.3,0.3,0.3), RGB(0.6,0.6,0.6), colors))}}) +
            @vlplot({:text, align = :left, dy = -10, dx = 5},
                transform = [{filter = "datum.condition == 'global'"}],
                x = :condition,
                y = :lower,
                text = :salience_label,
                color = :salience_label
            ) +
            @vlplot(:rule, x = :condition, y = :lower, y2 = :upper, color = :condition)
        )
    ) |>
    save(joinpath(dir, "raw_behavior_salience.svg"))

# Difference in hit rate across salience levels
# -----------------------------------------------------------------

diffmeans = @_ indmeans |>
    unstack(__, [:condition, :sid], :salience_label, :hits) |>
    transform!(__, [:high, :low] => (-) => :meandiff) |>
    groupby(__, :condition) |>
    combine(__,
        :meandiff => mean => :meandiff,
        :meandiff => lowerboot => :lower,
        :meandiff => upperboot => :upper
    )

barwidth = 18
@_ diffmeans |>
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
        color = {:condition, scale = {range = "#".*hex.(colors)}},
        y = {:meandiff,
            title = "High - Low Salience (Hit Rate)"
        }
    ) +
    @vlplot(:errorbar,
        x = {:condition, title = ""},
        y = {:lower, title = ""}, y2 = :upper,
    ) |> save(joinpath(dir, "raw_behavior_diff_salience.svg"))

# Difference in hit rate across salience levels (merve's summaries)
# -----------------------------------------------------------------

summaries = CSV.read(joinpath(processed_datadir("behavioral", "merve_summaries"), "export_salience.csv"))

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
pl |> save(joinpath(dir, "raw_sum_behavior_salience.svg"))

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
@_ diffmeans |>
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
        color = {:condition, scale = {range = "#".*hex.(colors)}},
        y = {:meandiff,
            title = "High - Low Salience (Hit Rate)"
        }
    ) +
    @vlplot(:errorbar,
        x = {:condition, title = ""},
        y = {:lower, title = ""}, y2 = :upper,
    ) |> save(joinpath(dir, "raw_sum_behavior_diff_salience.svg"))

# EEG Analysis
# =================================================================

# TODO: insert needed data loading

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

