
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

