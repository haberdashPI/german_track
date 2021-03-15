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

# Main figure
# =================================================================

# variable setup
# -----------------------------------------------------------------

prefix = joinpath(processed_datadir("analyses", "decode-timeline"), "testing")
GermanTrack.@load_cache prefix timelines

# setup plot data
plotdf = @_ timelines |>
    # @where(__, :train_type .!= "atmiss-target") |>
    # @where(__, :train_condition .== :condition) |>
    groupby(__, [:condition, :time, :sid, :is_target_source, :trial, :sound_index, :fold]) |>
    @combine(__, score = mean(:score))

# labels = OrderedDict(
#     "athit-target" => "Target Source",
#     "athit-other" => "Other Sources",
#     "athit-pre-target" => "Baseline"
# )
# tolabel(x) = labels[x]
tcolors = ColorSchemes.imola[[0.3, 0.8]]

# Paper plots
# -----------------------------------------------------------------

steps = range(0.1, 0.9, length = 15)
steps = vcat(steps[1] - step(steps), steps)
pcolors = ColorSchemes.batlow[steps[vcat(1,[1,7,12,14].+1)]]
pcolors[[1,end]] = GermanTrack.grayify.(pcolors[[1,end]])

target_len_y = -0.075
pl = @_ plotdf |>
    groupby(__, [:condition, :time, :is_target_source, :sid]) |>
    @combine(__, score = mean(:score)) |>
    @transform(__,
        time = :time .+ params.test.winlen_s,
        train_type =
            getindices(Dict(true => "Target Source", false => "Other Sources"),
            :is_target_source)
    ) |>
    # @where(__, -1 .< :time .< 2.5) |>
    groupby(__, [:condition, :time, :train_type]) |>
    combine(__, :score => boot(alpha = sqrt(0.05)) => AsTable) |>
    transform(__, [:condition, :train_type] =>
        ByRow((cond, type) -> type == "Other Sources" ? "other" :
            type == "Baseline" ? "before" : cond) => :train_label) |>
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
            color = {:train_label, sort = ["other", "global", "spatial", "object", "before"],
                title = "Source", scale = { range = "#".*hex.(pcolors) }}
        ) +
        @vlplot({:line, strokeJoin = :round},
            strokeDash = {:train_type, range = [[1,0], [4,1], [2,1]], sort = ["Target Source", "Other Sources", "Baseline"]},
            y = {:value, title = "Decoding Correlation"}) +
        @vlplot(:errorband, y = {:lower, title = ""}, y2 = :upper) +
        @vlplot({:text, align = "right", dx = -3, dy = -20},
            transform = [
                {filter = "datum.time > 2.1 && datum.time < 2.2 && datum.train_type == 'Target Source'"},
                {calculate = "split(datum.train_type,' ')", as = "train_type_lbl"}
            ],
            x = {:time, aggregate = :max, title = ""},
            y = {:upper, aggregate = :mean},
            text = :train_type_lbl
            # color = {value = "black"}
        ) +
        @vlplot({:text, align = "left", baseline = "top", dx = 0, dy = 9},
            transform = [
                {filter = "datum.time > 2 && datum.time < 2.1 && datum.train_type == 'Other Sources'"},
                {calculate = "split(datum.train_type,' ')", as = "train_type_lbl"}
            ],
            x = {:time, aggregate = :max, title = ""},
            y = {:lower, aggregate = :mean, scale = {domain = [-0.1, 0.15]}},
            text = :train_type_lbl
            # color = {value = "black"}
        ) +
        @vlplot({:text, align = "center", baseline = "top", dx = 3, dy = 3},
            transform = [
                {filter = "datum.time > 1.8 && datum.time < 1.9 && datum.train_type == 'Baseline'"},
                {calculate = "split(datum.train_type,' ')", as = "train_type_lbl"}
            ],
            x = {:time, aggregate = :max, title = ""},
            y = {:lower, aggregate = :mean, scale = {domain = [-0.1, 0.15]}},
            text = :train_type_lbl
            # color = {value = "black"}
        )
    ) +
    # Target annotation
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:text, size = 11, baseline = "bottom", align = :center},
            x = {datum = 0.5}, y = {datum = target_len_y},
            text = {value = ["Target", "Extent"]},
            color = {value = "black"}
        ) +
        @vlplot({:rect, opacity = 0.25},
            x = {datum = 0}, x2 = {datum = 1},
            color = {value = "gray"},
        )
    ));
pl |> save(joinpath(dir, "fig2c.svg"))

# Supplement: decoding difference
# =================================================================

target_len_y = -0.075
pl = @_ plotdf |>
    @transform(__,
        train_type = train_type = getindices(Dict(true => "target", false => "other"),
            :is_target_source)
    ) |>
    groupby(__, [:condition, :time, :train_type, :sid]) |>
    @combine(__, score = mean(:score)) |>
    unstack(__, [:condition, :time, :sid], :train_type, :score) |>
    @transform(__,
        time = :time .+ params.test.winlen_s,
        diff = :target .- :other,
    ) |>
    # @where(__, -1 .< :time .< 2.5) |>
    groupby(__, [:condition, :time]) |>
    combine(__, :diff => boot(alpha = sqrt(0.05)) => AsTable) |>
    @vlplot(
        spacing = 5,
        config = {legend = {disable = true}},
    ) + (@vlplot() +
    (
        @vlplot(
            width = 128, height = 130,
            x = {:time, type = :quantitative, title = "Time (s)"},
            color = {:condition, sort = ["global", "spatial", "object", "before"],
                title = "Source", scale = { range = "#".*hex.(GermanTrack.colors) }}
        ) +
        @vlplot({:line, strokeJoin = :round},
            y = {:value, title = "Target - Other Correlation"}) +
        @vlplot(:errorband, y = {:lower, title = ""}, y2 = :upper)
    ));
pl |> save(joinpath(dir, "decode_diff.svg"))

# Supplement: decoding from start by source
# =================================================================

prefix = joinpath(processed_datadir("analyses", "decode-timeline-source"), "testing")
GermanTrack.@load_cache prefix timelines

# two figures:
# - trained sources when they're the target, across the different lags
# - trained sources for target - non-target cases (for each lag at first, but probalby just for the shortest lag)

# Timeline for decoding by source, when the soure is the target
# -----------------------------------------------------------------

pcolors = ColorSchemes.imola[range(0.2,0.8,length=3)]

# setup plot data
plotdf = @_ timelines |>
    @where(__, :is_target_source .& (:source .== :trained_source)) |>
    groupby(__, [:condition, :time, :sid, :trial, :sound_index, :fold, :lagcut]) |>
    @combine(__, score = mean(:score))

laglabels = Dict(
    0 => "3 sec",
    32 => "2 sec",
    64 => "1 sec"
)

target_len_y = -0.075
pl = @_ plotdf |>
    groupby(__, [:condition, :time, :lagcut, :sid]) |>
    @combine(__, score = mean(:score)) |>
    # @where(__, -1 .< :time .< 2.5) |>
    groupby(__, [:condition, :time, :lagcut]) |>
    combine(__, :score => boot(alpha = sqrt(0.05)) => AsTable) |>
    @transform(__, laglabel = getindices(laglabels, :lagcut)) |>
    @vlplot(
        spacing = 5,
        # config = {legend = {disable = true}},
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
            color = {:laglabel, type = "ordinal",
                title = "Lags", scale = { range = "#".*hex.(pcolors) }}
        ) +
        @vlplot({:line, strokeJoin = :round},
            # strokeDash = {:test_type, range = [[1,0], [4,1], [2,1]], sort = ["Trained Source", "Other Sources", "Baseline"]},
            y = {:value, title = "Decoding Correlation"}) +
        @vlplot(:errorband, y = {:lower, title = ""}, y2 = :upper)
    ));
pl |> save(joinpath(dir, "decode_by_source_trained_target.svg"))

# Decode by source, difference from target vs non-target trials
# -----------------------------------------------------------------

pcolors = GermanTrack.colors

# setup plot data
plotdf = @_ timelines |>
    @where(__, (:source .== :trained_source) .& (:lagcut .== 64)) |>
    groupby(__, [:condition, :time, :sid, :trial, :sound_index, :fold, :lagcut, :is_target_source]) |>
    @combine(__, score = mean(:score)) |>
    groupby(__, Not([:trial])) |>
    @combine(__, score = mean(:score)) |>
    @transform(__, source_kind = ifelse.(:is_target_source, "target", "other")) |>
    select(__, Not(:is_target_source)) |>
    unstack(__, Not([:source_kind, :score]), :source_kind, :score) |>
    @transform(__, diff = :target .- :other) |>
    groupby(__, [:condition, :time, :lagcut, :sid]) |>
    @combine(__, diff = mean(:diff)) |>
    groupby(__, [:condition, :time, :lagcut]) |>
    combine(__, :diff => boot(alpha = sqrt(0.05)) => AsTable) |>
    @transform(__, laglabel = getindices(laglabels, :lagcut))

target_len_y = -0.075
pl = @_ plotdf |>
    @transform(__, laglabel = getindices(laglabels, :lagcut)) |>
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
            color = {:condition, type = "ordinal",
                title = "Source", scale = { range = "#".*hex.(pcolors) }}
        ) +
        @vlplot({:line, strokeJoin = :round},
            # strokeDash = {:test_type, range = [[1,0], [4,1], [2,1]], sort = ["Trained Source", "Other Sources", "Baseline"]},
            y = {:value, title = "Target - Non-target"}) +
        @vlplot(:errorband, y = {:lower, title = ""}, y2 = :upper)
    ));
pl |> save(joinpath(dir, "decode_by_source_target_diff.svg"))

# Supplement 0: decoding broken down by early/late near/far
# and their interactions
# =================================================================

steps = range(0.1, 0.9, length = 15)
steps = vcat(steps[1] - step(steps), steps)
pcolors = ColorSchemes.batlow[steps[vcat(1,[1,7,12].+1)]]
pcolors[1] = GermanTrack.grayify(pcolors[1])

# setup plot data
plotdf = @_ timelines |>
    @where(__, :train_type .!= "atmiss-target") |>
    groupby(__, Not([:encoding, :source])) |>
    @combine(__, score = mean(:score)) |>
    @transform(__,
        target_time_label = getindex.(Ref(meta.target_time_label), :sound_index),
        target_switch_label = getindex.(Ref(meta.target_switch_label), :sound_index)
    )

target_len_y = 0.15
pl = @_ plotdf |>
    groupby(__, [:condition, :time, :train_type, :sid, :target_switch_label]) |>
    @combine(__, score = mean(:score)) |>
    @transform(__,
        time = :time .+ params.test.winlen_s,
        train_type = tolabel.(:train_type)
    ) |>
    @where(__, -1 .< :time .< 2.5) |>
    groupby(__, [:condition, :time, :train_type, :target_switch_label]) |>
    combine(__, :score => boot => AsTable) |>
    transform(__, [:condition, :train_type] =>
        ByRow((cond, type) -> type == "Other Sources" ? "other" : cond) => :train_label) |>
    @vlplot(
        spacing = 5,
        config = {legend = {disable = true}},
    facet = {
        row = {field = :target_switch_label, sort = ["near", "far"]},
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
            y = {:lower, aggregate = :mean, scale = {domain = [-0.1, 0.2]}},
            text = :train_type_lbl
            # color = {value = "black"}
        )
    ) +
    # Target annotation
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:text, size = 11, baseline = "bottom", align = :center},
            x = {datum = 0.5}, y = {datum = target_len_y},
            text = {value = ["Target", "Extent"]},
            color = {value = "black"}
        ) +
        @vlplot({:rect, opacity = 0.25},
            x = {datum = 0}, x2 = {datum = 1},
            color = {value = "gray"},
        )
    ));
pl |> save(joinpath(dir, "decode_nearfar_condition.svg"))

target_len_y = 0.2
pl = @_ plotdf |>
    groupby(__, [:condition, :time, :train_type, :sid, :target_time_label]) |>
    @combine(__, score = mean(:score)) |>
    @transform(__,
        time = :time .+ params.test.winlen_s,
        train_type = tolabel.(:train_type)
    ) |>
    @where(__, -1 .< :time .< 2.5) |>
    groupby(__, [:condition, :time, :train_type, :target_time_label]) |>
    combine(__, :score => boot => AsTable) |>
    transform(__, [:condition, :train_type] =>
        ByRow((cond, type) -> type == "Other Sources" ? "other" : cond) => :train_label) |>
    @vlplot(
        spacing = 5,
        config = {legend = {disable = true}},
    facet = {
        row = {field = :target_time_label, sort = ["early", "late"]},
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
            y = {:lower, aggregate = :mean, scale = {domain = [-0.15, 0.25]}},
            text = :train_type_lbl
            # color = {value = "black"}
        )
    ) +
    # Target annotation
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:text, size = 11, baseline = "bottom", align = :center},
            x = {datum = 0.5}, y = {datum = target_len_y},
            text = {value = ["Target", "Extent"]},
            color = {value = "black"}
        ) +
        @vlplot({:rect, opacity = 0.25},
            x = {datum = 0}, x2 = {datum = 1},
            color = {value = "gray"},
        )
    ));
pl |> save(joinpath(dir, "decode_earlylate_condition.svg"))

target_len_y = 0.2
pl = @_ plotdf |>
    @where(__, :condition .== "object") |>
    groupby(__, [:condition, :time, :train_type, :sid, :target_switch_label, :target_time_label]) |>
    @combine(__, score = mean(:score)) |>
    @transform(__,
        time = :time .+ params.test.winlen_s,
        train_type = tolabel.(:train_type)
    ) |>
    @where(__, -1 .< :time .< 2.5) |>
    groupby(__, [:condition, :time, :train_type, :target_switch_label, :target_time_label]) |>
    combine(__, :score => boot => AsTable) |>
    transform(__, [:condition, :train_type] =>
        ByRow((cond, type) -> type == "Other Sources" ? "other" : cond) => :train_label) |>
    @vlplot(
        spacing = 5,
        config = {legend = {disable = true}},
    facet = {
        row = {field = :target_time_label, sort = ["early", "late"]},
        column = {field = :target_switch_label, sort = ["near", "far"]}
    }) + (@vlplot() +
    (
        @vlplot(
            width = 128, height = 130,
            x = {:time, type = :quantitative, title = "Time (s)"},
            color = {:train_label, sort = ["other", "object"],
                title = "Source", scale = { range = "#".*hex.(pcolors[[1,4]]) }}
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
            y = {:lower, aggregate = :mean, scale = {domain = [-0.15, 0.25]}},
            text = :train_type_lbl
            # color = {value = "black"}
        )
    ) +
    # Target annotation
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:text, size = 11, baseline = "bottom", align = :center},
            x = {datum = 0.5}, y = {datum = target_len_y},
            text = {value = ["Target", "Extent"]},
            color = {value = "black"}
        ) +
        @vlplot({:rect, opacity = 0.25},
            x = {datum = 0}, x2 = {datum = 1},
            color = {value = "gray"},
        )
    ));
pl |> save(joinpath(dir, "decode_earlylate_nearfar.svg"))

# Supplement 1: decoding generalization across conditions
# =================================================================

# setup plot data
plotdf = @_ timelines |>
    groupby(__, [:condition, :train_condition, :time, :sid, :train_type, :trial, :sound_index, :fold]) |>
    @combine(__, score = mean(:score))

labels = OrderedDict(
    "athit-target" => "Target Source",
    "athit-other" => "Other Sources",
)
tolabel(x) = labels[x]
tcolors = ColorSchemes.imola[[0.3, 0.8]]

supdir = mkpath(joinpath(dir, "supplement"))

steps = range(0.1, 0.9, length = 15)
steps = vcat(steps[1] - step(steps), steps)
pcolors = ColorSchemes.batlow[steps[vcat(1,[1,7,12].+1)]]
pcolors[1] = GermanTrack.grayify(pcolors[1])

target_len_y = 0.135
pl = @_ plotdf |>
    groupby(__, [:condition, :train_condition, :time, :train_type, :sid]) |>
    @combine(__, score = mean(:score)) |>
    @transform(__,
        time = :time .+ params.test.winlen_s,
        train_type = tolabel.(:train_type)
    ) |>
    @where(__, -1 .< :time .< 2.5) |>
    groupby(__, [:condition, :train_condition, :time, :train_type]) |>
    combine(__, :score => boot => AsTable) |>
    transform(__, [:condition, :train_type] =>
        ByRow((cond, type) -> type == "Other Sources" ? "other" : cond) => :train_label) |>
    @vlplot(
        spacing = 5,
        config = {legend = {disable = true}},
    facet = {
        row = {
            field = :train_condition, title = "Decoder Training",
            sort = ["global", "spatial", "object"],
                header = {labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"},
        },
        column = {field = :condition, title = "Decoder Testing",
            sort = ["global", "spatial", "object"],
            header = {
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)",
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
pl |> save(joinpath(supdir, "decode_condition_generalize.svg"))

# Supplement 2: decoding by miss
# =================================================================

# setup plot data
plotdf = @_ timelines |>
    @where(__, :train_type .!= "athit-other") |>
    @where(__, :condition .== :train_condition) |>
    groupby(__, [:condition, :time, :sid, :train_type, :trial, :sound_index, :fold]) |>
    @combine(__, score = mean(:score))

labels = OrderedDict(
    "athit-target" => "Hit",
    "atmiss-target" => "Miss",
)
tolabel(x) = labels[x]
tcolors = ColorSchemes.imola[[0.3, 0.8]]

supdir = mkpath(joinpath(dir, "supplement"))

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
        ByRow((cond, type) -> type == "Miss" ? "miss" : cond) => :train_label) |>
    @vlplot(
        spacing = 5,
        config = {legend = {disable = true}},
    facet = {
        column = {field = :condition, title = "Decoder Testing",
            sort = ["global", "spatial", "object"],
            header = {
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)",
            }
        }
    }) + (@vlplot() +
    (
        @vlplot(
            width = 128, height = 130,
            x = {:time, type = :quantitative, title = "Time (s)"},
            color = {:train_label, sort = ["miss", "global", "spatial", "object"],
                title = "Source", scale = { range = "#".*hex.(pcolors) }}
        ) +
        @vlplot({:line, strokeJoin = :round}, y = {:value, title = "Decoding Correlation"}) +
        @vlplot(:errorband, y = {:lower, title = ""}, y2 = :upper) +
        @vlplot({:text, align = "left", dx = 3, dy = -9},
            transform = [
                {filter = "datum.time > 1.25 && datum.time < 1.5 && datum.train_type == 'Hit'"},
                {calculate = "split(datum.train_type,' ')", as = "train_type_lbl"}
            ],
            x = {:time, aggregate = :max, title = ""},
            y = {:upper, aggregate = :mean},
            text = :train_type_lbl
            # color = {value = "black"}
        ) +
        @vlplot({:text, align = "right", baseline = "top", dx = 3, dy = 3},
            transform = [
                {filter = "datum.time > 1.3 && datum.time < 1.4 && datum.train_type == 'Miss'"},
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
pl |> save(joinpath(supdir, "decode_condition_miss.svg"))


# Supplement 3: decoding fall outside CI of baseline
# =================================================================



# Presentation plots
# =================================================================

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

