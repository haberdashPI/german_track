# Setup
# =================================================================

using DrWatson #; @quickactivate("german_track")
using EEGCoding, GermanTrack, DataFrames, StatsBase, Underscores, Transducers,
    BangBang, ProgressMeter, HDF5, DataFramesMeta, Lasso, VegaLite, Colors,
    Printf, LambdaFn, ShiftedArrays, ColorSchemes, Flux, CUDA, GLM, SparseArrays,
    JLD, Arrow, FFTW, GLM, CategoricalArrays, Tables, DataStructures

dir = mkpath(joinpath(plotsdir(), "figure2_parts"))

using GermanTrack: colors

include(joinpath(scriptsdir(), "julia", "setup_decode_params.jl"))
# NOTE: these area parameters copied from process_decode_timelilne
# should be come parameters

# variable setup
# -----------------------------------------------------------------

prefix = joinpath(processed_datadir("analyses", "decode-timeline"), "testing")
GermanTrack.@load_cache prefix timelines

# setup plot data
plotdf = @_ timelines |>
    groupby(__, [:condition, :time, :sid, :train_type, :trial, :sound_index, :fold]) |>
    @combine(__, score = mean(:score))

labels = OrderedDict(
    "athit-target" => "Target Source",
    "athit-other" => "Other Sources",
)
tolabel(x) = labels[x]
tcolors = ColorSchemes.imola[[0.3, 0.8]]

# Paper plots
# -----------------------------------------------------------------

steps = range(0.1, 0.9, length = 15)
steps = vcat(steps[1] - step(steps), steps)
pcolors = ColorSchemes.batlow[steps[vcat(1,[1,7,12].+1)]]
pcolors[1] = GermanTrack.grayify(pcolors[1])

target_len_y = 0.135
pl = @_ plotdf |>
    groupby(__, [:condition, :time, :train_type, :sid]) |>
    @combine(__, score = mean(:score)) |>
    @transform(__,
        time = :time .+ params.test.winlen_s,
        train_type = tolabel.(:train_type)
    ) |>
    @where(__, -1 .< :time .< 2.5) |>
    groupby(__, [:condition, :time, :train_type]) |>
    combine(__, :score => boot => AsTable) |>
    transform(__, [:condition, :train_type] =>
        ByRow((cond, type) -> type == "Other Sources" ? "other" : cond) => :train_label) |>
    @vlplot(
        spacing = 5,
        config = {legend = {disable = true}},
    facet = {
        column = {field = :condition, title = "",
            sort = ["global", "spatial", "object"],
            header = {
                title = "",
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)",
                labelFontWeight = "bold",
            }
        }
    }) + (@vlplot() +
    (
        @vlplot(
            width = 128, height = 130,
            x = {:time, type = :quantitative, title = "Time (s)"},
            color = {:train_label, sort = ["other", "global", "spatial", "object"],
                title = "Source", scale = { range = "#".*hex.(pcolors) }}
        ) +
        @vlplot({:line, strokeJoin = :round}, y = {:value, title = "Decoding Correlation"}) +
        @vlplot(:errorband, y = {:lower, title = ""}, y2 = :upper) +
        @vlplot({:text, align = "left", dx = 3, dy = -9},
            transform = [
                {filter = "datum.time > 1.25 && datum.time < 1.5 && datum.train_type == 'Target Source'"},
                {calculate = "split(datum.train_type,' ')", as = "train_type_lbl"}
            ],
            x = {:time, aggregate = :max, title = ""},
            y = {:upper, aggregate = :mean},
            text = :train_type_lbl
            # color = {value = "black"}
        ) +
        @vlplot({:text, align = "right", baseline = "top", dx = 3, dy = 3},
            transform = [
                {filter = "datum.time > 1.3 && datum.time < 1.4 && datum.train_type == 'Other Sources'"},
                {calculate = "split(datum.train_type,' ')", as = "train_type_lbl"}
            ],
            x = {:time, aggregate = :max, title = ""},
            y = {:lower, aggregate = :mean, scale = {domain = [-0.05, 0.15]}},
            text = :train_type_lbl
            # color = {value = "black"}
        )

    ) +
    # Target annotation
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:text, size = 11, baseline = "bottom", align = :left},
            x = {datum = 0}, y = {datum = target_len_y},
            text = {value = ["Target", "Extent"]},
            color = {value = "black"}
        ) +
        @vlplot({:rect, opacity = 0.25},
            x = {datum = 0}, x2 = {datum = 1},
            color = {value = "gray"},
        )
    ));
pl |> save(joinpath(dir, "fig2c.svg"))

# Presentation plots
# -----------------------------------------------------------------

mkpath(joinpath(dir, "present"))
thresh = 0.1

pldata = @_ timelines |>
    groupby(__, Not([:encoding, :source])) |>
    @combine(__, score = mean(:score)) |>
    @transform(__,
        time = :time .+ params.test.winlen_s,
        train_type = tolabel.(:train_type),
    ) |>
    @where(__, (-2 .< :time .< -1) .| (0.5 .< :time .< 1.5)) |>
    @transform(__,
        region_label = ifelse.(:time .< 0, "before-target", "near-target"),
    ) |>
    unstack(__, Not([:train_type, :score]), :train_type, :score, allowduplicates = true) |>
    groupby(__, [:condition, :region_label, :sid]) |>
    @combine(__,
        favor_target = mean((:var"Target Source" .- :var"Other Sources") .> thresh),
        favor_other = mean((:var"Target Source" .- :var"Other Sources") .< -thresh)
    ) |>
    stack(__, r"favor", variable_name = :favor) |>
    @transform(__, favor = replace.(:favor, r"favor_(.*)" => s"\1")) |>
    groupby(__, [:condition, :region_label, :favor]) |>
    combine(__, :value => boot(alpha = sqrt(0.05)) => AsTable)

barwidth = 18
pl = @_ pldata |>
    @transform(__,
        region_label = replace(:region_label,
            "near-target" => "Near (0.5 s, 1.5 s)",
            "before-target" => "Before (-2 s, -1 s)"
        )
    ) |>
    @vlplot(spacing = 5,
        config = {
            legend = {disable = true},
            bar = {discreteBandSize = barwidth},
        },
        facet = {
            row = {field = :region_label, title = "", sort = ["Near (0.5 s, 1.5 s)", "Before (-2 s, -1 s)"]},
            column = {field = :condition, title = "", sort = ["global", "spatial", "object"],
                header = {labelFontWeight = "bold",
                    labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}
            }
        }) +
    (@vlplot(x = {:favor, axis = {title = "Source", labelAngle = -45}},
        height = 90, width = 60) +
    @vlplot({:bar}, y = {:value, title = "Time Favoring (s)"}, color = :condition) +
    @vlplot(:rule, y = :lower, y2 = :upper));
pl |> save(joinpath(dir, "present", "fig2d.svg"))

pl = @_ plotdf |>
    groupby(__, [:condition, :time, :train_type, :sid]) |>
    @combine(__, score = mean(:score)) |>
    @transform(__,
        time = :time .+ params.test.winlen_s,
        train_type = tolabel.(:train_type)
    ) |>
    @where(__, -1 .< :time .< 2.5) |>
    groupby(__, [:condition, :time, :train_type]) |>
    combine(__, :score => boot => AsTable) |>
    transform(__, [:condition, :train_type] =>
        ByRow((cond, type) -> type == "Other Sources" ? "other" : cond) => :train_label) |>
    @vlplot(
        spacing = 5,
        config = {legend = {disable = true}},
    facet = {
        column = {field = :condition, title = "",
            sort = ["global", "spatial", "object"],
            header = {
                title = "",
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)",
                labelFontWeight = "bold",
            }
        }
    }) +
    (
        @vlplot(
            width = 60, height = 80,
            x = {:time, type = :quantitative, title = ""},
            color = {:train_label, sort = ["other", "global", "spatial", "object"],
                title = "Source", scale = { range = "#".*hex.(pcolors) }}
        ) +
        @vlplot({:line, strokeJoin = :round}, y = {:value, title = "Mean Correlation"}) +
        @vlplot(:errorband, y = {:lower, title = ""}, y2 = :upper)
    );
pl |> save(joinpath(dir, "decode_timeline.svg"))


tcolors = ColorSchemes.lajolla[range(0.3,0.9, length = 4)]
pl = @_ plotdf |>
    groupby(__, [:condition, :time, :train_type, :sid]) |>
    @combine(__, score = mean(:score)) |>
    @transform(__, time = :time .+ params.test.winlen_s,
        condition = uppercasefirst.(:condition)) |>
    @where(__, -1 .< :time .< 2.5) |>
    unstack(__, [:condition, :time, :sid], :train_type, :score) |>
    @transform(__, scorediff = :var"athit-target" .- :var"athit-other") |> #"
    groupby(__, [:condition, :time]) |>
    combine(__, :scorediff => boot(alpha = sqrt(0.05)) => AsTable) |>
    @vlplot(
        config = {legend = {disable = true}},
        height = 90, width = 100,
    ) +
    (
        @vlplot(x = {:time, type = :quantitative, title = "Time"}, color = {:condition, scale = {range = "#".*hex.(colors)}}) +
        @vlplot({:line, strokeJoin = :round}, y = {:value, title = ["Target - Other", "Correlation"]}) +
        @vlplot(:errorband, y = {:lower, title = ""}, y2 = :upper) +
        @vlplot({:text, align = "left", dx = 3},
            transform = [
                {filter = "datum.time > 2.25 && datum.time < 2.5"},
            ],
            x = {:time, aggregate = :max, title = ""},
            y = {:value, aggregate = :mean},
            text = {field = :condition}
            # color = {value = "black"}
        )
    );
pl |> save(joinpath(dir, "decode_timeline_diff.svg"))

