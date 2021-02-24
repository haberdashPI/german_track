# Setup
# =================================================================

using DrWatson #; @quickactivate("german_track")
using EEGCoding, GermanTrack, DataFrames, StatsBase, Underscores, Transducers,
    BangBang, ProgressMeter, HDF5, DataFramesMeta, Lasso, VegaLite, Colors,
    Printf, LambdaFn, ShiftedArrays, ColorSchemes, Flux, CUDA, GLM, SparseArrays,
    JLD, Arrow, FFTW, GLM, CategoricalArrays, Tables, DataStructures

dir = mkpath(joinpath(plotsdir(), "figure3_parts"))

using GermanTrack: colors

include(joinpath(scriptsdir(), "julia", "setup_decode_params.jl"))
# NOTE: these area parameters copied from process_decode_timelilne
# should be come parameters

# variable setup
# -----------------------------------------------------------------

prefix = joinpath(processed_datadir("analyses", "decode"), "train")
GermanTrack.@load_cache prefix predictions_

tcolors = ColorSchemes.lajolla[[0.3,0.9]]
mean_offset = 15
ind_offset = 6
pl = @_ predictions_ |>
    @where(__, :train_type .!= "atmiss-target") |>
    @transform(__, score = cor.(:prediction, :data)) |>
    groupby(__, Not(:encoding)) |>
    @combine(__, score = mean(:score)) |>
    groupby(__, [:sid, :condition, :train_type, :source, :salience]) |>
    @combine(__, score = mean(:score)) |>
    groupby(__, [:sid, :condition, :train_type, :salience]) |>
    @combine(__, score = mean(:score)) |>
    @transform(__,
        train_type = replace(:train_type,
            "athit-other" => "Other Sources",
            "athit-target" => "Target Source"
        )
    ) |>
    @vlplot(config = {legend = {disable = true}},
        width = 130, height = 100,
    ) + (
        @vlplot({:point, filled = true, opacity = 0.6},
            x     = :salience,
            y     = {:score, type = :quantitative, aggregate = :mean},
            color = {:train_type, scale = {range = "#".*hex.(tcolors)}})
        ) +
        @vlplot(:line,
            transform = [{regression = :score, on = :salience, groupby = [:train_type]}],
            color = :train_type,
            x = :salience, y = :score
        ) +
        @vlplot({:text, align = "left", angle = 0, dx = 8, dy = -8, baseline = "bottom"},
            transform = [
                {regression = :score, on = :salience, groupby = [:train_type]},
                {filter = "datum.salience > 0"},
                {filter = "datum.train_type == 'Target Source'"},
                {calculate = "split(datum.train_type,' ')", as = :train_type_label}
            ],
            x = {:salience, aggregate = :maximum, title = "Target Salience"},
            y = {:score, aggregate = :maximum, title = "Decoding Correlation"},
            text = :train_type_label, color = :train_type
        ) +
        @vlplot({:text, align = "left", angle = 0, dx = 8, dy = 0, baseline = "top"},
            transform = [
                {regression = :score, on = :salience, groupby = [:train_type]},
                {filter = "datum.salience > 0"},
                {filter = "datum.train_type != 'Target Source'"},
                {calculate = "split(datum.train_type,' ')", as = :train_type_label}
            ],
            x = {:salience, aggregate = :maximum},
            y = {:score, aggregate = :maximum, title = "Decoding Correlation"},
            text = :train_type_label, color = :train_type
        );
pl |> save(joinpath(dir, "fig3e.svg"))

