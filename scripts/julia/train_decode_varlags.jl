# Setup
# =================================================================

using DrWatson
@quickactivate("german_track")
println("Julia version: $VERSION.")

using EEGCoding, GermanTrack, DataFrames, StatsBase, Underscores, Transducers,
    BangBang, ProgressMeter, HDF5, DataFramesMeta, Lasso, VegaLite, Colors,
    Printf, ShiftedArrays, ColorSchemes, Flux, CUDA, GLM, SparseArrays,
    JLD, Arrow, FFTW, GLM, CategoricalArrays, Tables, DataStructures,
    PooledArrays # (pooled arrays is needed to reload subject data)

dir = processed_datadir("analyses", "decode", "plots")
include(joinpath(scriptsdir(), "julia", "setup_decode_params.jl")) # defines `params`

x, nfeatures = prepare_decode_data(params)
stimulidf = prepare_decode_stimuli(params)

cutlags(x, nfeatures, ncut) = @view x[:, (ncut*nfeatures+1):end]

# Train Model
# =================================================================

@info "Cross-validated training of source decoders (this will take some time...)"

train_types = OrderedDict(
    # "athit-other" => ((df, kind) -> @where(df,
    #     (:hittype .== "hit") .&
    #     (:windowing .== "target") .&
    #     :is_target_source)),
    # "athit-other" => ((df, kind) -> @where(df,
    #     (:hittype .== "hit") .&
    #     (:windowing .== "target") .&
    #     .!(:is_target_source))),
    # "athit-pre-target" => ((df, kind) -> @where(df,
    #     (:hittype .== "hit") .&
    #     (:windowing .== "pre-target") .&
    #     :is_target_source)),
    "random" => ((df, kind) -> @where(df,
        (:hittype .∈ Ref(["hit", "miss"])) .&
        contains.(:windowing, "random"))
    )
)

modelsetup = @_ stimulidf |>
    @where(__, :condition .!= "spatial") |>
    groupby(__, [:condition, :source, :encoding]) |>
    repeatby(__,
        :lagcut => [1.0, 2.0],
        :cross_fold => 1:params.train.nfolds,
        :λ => params.train.λs,
        :train_type => keys(train_types)) |>
    testsplit(__, :sid, rng = df -> stableRNG(2019_11_18, :validate_flux,
        NamedTuple(df[1, [:cross_fold, :λ, :train_type, :encoding]])))

train_decoder(params, x, modelsetup, train_type)

predictions = @_ modelrun |> groupby(__, Not([:result, :model, :validate])) |>
    combine(only(_.result), __)
valpredictions = @_ modelrun |> groupby(__, Not([:result, :model, :validate])) |>
    combine(only(_.validate), __)
models = select(modelrun, Not([:result, :validate]))

best_λ = plot_decode_lamabds(predictions)

# Store only the best results
# -----------------------------------------------------------------

models_ = @_ filter(_.λ == best_λ[_.cross_fold], models)
predictions_ = @_ filter(_.λ == best_λ[_.cross_fold], predictions)

prefix = joinpath(processed_datadir("analyses", "decode"), "train")
GermanTrack.@save_cache prefix (models_, :bson) predictions_
