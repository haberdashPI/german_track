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

dir = processed_datadir("analyses", "decode-varlag", "plots")
include(joinpath(scriptsdir(), "julia", "setup_decode_params.jl")) # defines `params`

prefix = joinpath(processed_datadir("analyses", "decode-varlag"), "train")

x, windows, nfeatures = prepare_decode_data(params, prefix)
stimulidf = prepare_decode_stimuli(params, windows, prefix)

cutlags(x, nfeatures, ncut) = (ncut*nfeatures+1):size(x, 1)

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
    "random" => MultiSelector(
        StimSelector((df, kind) -> @where(df,
            (:hittype .∈ Ref(["hit", "miss"])) .&
            contains.(:windowing, "random"))),
        EEGFeatureSelector((x, row) -> cutlags(x, nfeatures, row.lagcut))
    )
)

modelsetup = @_ stimulidf |>
    @where(__, :condition .== "global") |>
    @where(__, :target_source .== "male") |>
    @where(__, :condition .!= "spatial") |>
    groupby(__, [:condition, :source, :encoding]) |>
    repeatby(__,
        :lagcut => round.(Int, [1.0, 2.0] * params.stimulus.samplerate),
        :cross_fold => 1:params.train.nfolds,
        :λ => params.train.λs,
        :train_type => keys(train_types)) |>
    testsplit(__, :sid, rng = df -> stableRNG(2019_11_18, :validate_flux,
        NamedTuple(df[1, [:cross_fold, :λ, :train_type, :encoding]])))

predictions, valpredictions, models = train_decoder(params, x, modelsetup, train_types)

best_λ = plot_decode_lambdas(params, predictions, valpredictions, dir)

# Store only the best results
# -----------------------------------------------------------------

models_ = @_ filter(_.λ == best_λ[_.cross_fold], models)
predictions_ = @_ filter(_.λ == best_λ[_.cross_fold], predictions)

GermanTrack.@save_cache prefix (models_, :bson) predictions_
