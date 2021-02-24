# Setup
# =================================================================

using DrWatson #; @quickactivate("german_track")
using EEGCoding, GermanTrack, DataFrames, StatsBase, Underscores, Transducers,
    BangBang, ProgressMeter, HDF5, DataFramesMeta, Lasso, VegaLite, Colors,
    Printf, LambdaFn, ShiftedArrays, ColorSchemes, Flux, CUDA, GLM, SparseArrays,
    JLD, Arrow, FFTW, GLM, CategoricalArrays, Tables, DataStructures

dir = mkpath(joinpath(plotsdir(), "figure4_parts"))

using GermanTrack: colors

include(joinpath(scriptsdir(), "julia", "setup_decode_params.jl"))
# NOTE: these area parameters copied from process_decode_timelilne
# should be come parameters

# variable setup
# -----------------------------------------------------------------

prefix = joinpath(processed_datadir("analyses", "decode"), "train")
GermanTrack.@load_cache prefix predictions_

# Plotting
# =================================================================

tcolors = reverse(ColorSchemes.lajolla[[0.3,0.9]])
mean_offset = 15
ind_offset = 6
labels = OrderedDict(
    "athit-target" => "Target Source",
    "athit-other" => "Other Source",
)

tolabel(x) = labels[x]
timedf = @_ predictions_ |>
    @where(__, :train_type .!= "atmiss-target") |>
    @transform(__, score = cor.(:prediction, :data)) |>
    groupby(__, Not(:encoding)) |>
    @combine(__, score = mean(:score)) |>
    groupby(__, [:sid, :condition, :train_type, :source, :target_time]) |>
    @combine(__, score = mean(:score)) |>
    groupby(__, [:sid, :condition, :train_type, :target_time]) |>
    @combine(__, score = mean(:score)) |>
    @transform(__,
        condition = string.(:condition),
        train_type = tolabel.(:train_type),
    )

pl = timedf |>
    @vlplot(config = {legend = {disable = true}},
        width = 130, height = 110,
        # facet = {
        #     column = {field = :condition, type = :ordinal},
        #     # row = {field = :train_type, type = :ordinal}
        # }
    ) + (
        @vlplot({:point, filled = true, opacity = 0.6},
            x     = {:target_time, title = "Target Time (s)", scale = {zero = false, padding = 5}},
            y     = {:score, title = "Decoding Correlation", type = :quantitative, aggregate = :mean},
            color = {:train_type, scale = {range = "#".*hex.(tcolors)},
                sort = collect(values(labels)),
                title = ""}
        )
    ) + (
        @vlplot() +
        @vlplot(:line,
            color = {:train_type, sort = collect(values(labels))},
            transform = [{regression = :score, on = :target_time, groupby = [:train_type]}],
            x = :target_time, y = :score
        )
    ) + (
        @vlplot() +
        @vlplot({:text, align = "left", angle = 0, dx = 12, dy = -8, baseline = "bottom"},
            transform = [
                {regression = :score, on = :target_time, groupby = [:train_type]},
                {filter = "datum.target_time > 5"},
                {filter = "datum.train_type == 'Target Source'"},
                {calculate = "split(datum.train_type,' ')", as = :train_type_label}
            ],
            x = {:target_time, aggregate = :maximum, title = ""},
            y = {:score, aggregate = :maximum, title = ""},
            text = :train_type_label, color = {:train_type, sort = collect(values(labels))}
        ) +
        @vlplot({:text, align = "left", angle = 0, dx = 12, dy = 0, baseline = "top"},
            transform = [
                {regression = :score, on = :target_time, groupby = [:train_type]},
                {filter = "datum.target_time > 5"},
                {filter = "datum.train_type != 'Target Source'"},
                {calculate = "split(datum.train_type,' ')", as = :train_type_label}
            ],
            x = {:target_time, aggregate = :maximum},
            y = {:score, aggregate = :maximum, title = ""},
            text = :train_type_label, color = {:train_type, sort = collect(values(labels))}
        )
    );
pl |> save(joinpath(dir, "fig4c.svg"))
