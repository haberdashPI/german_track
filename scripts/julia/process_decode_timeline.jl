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

# STEPS: maybe we should consider cross validating across stimulus type
# rather than subject id?

# Load subject data
# -----------------------------------------------------------------

include(joinpath(scriptsdir(), "julia", "setup_decode_params.jl"))

subjects = load_decode_data()
# things that are duplicated in `train_decode`: should be defined in a common file
# and written out to a TOML parameter setup

decode_prefix = joinpath(processed_datadir("analyses", "decode"), "train")
GermanTrack.@load_cache decode_prefix (models_, :bson) stimulidf x_scores
meta = GermanTrack.load_stimulus_metadata()

# timeline testing
# -----------------------------------------------------------------

sid_trial = mapreduce(x -> x[1] .=> eachindex(x[2].eeg.data), vcat, pairs(subjects))
groups = groupby(stimulidf, [:sid, :trial, :condition])
progress = Progress(length(groups), desc = "Evaluating decoder over timeline...")

modelgroups = groupby(models_, [:source, :encoding, :cross_fold, :train_type])

timelines = combine(groups) do trialdf
    sid, trial, sound_index, target_time, fold =
        trialdf[:, [:sid, :trial, :sound_index, :target_time, :fold]] |>
        eachrow |> unique |> only

    trialdata = withlags(subjects[sid].eeg[trial]', params.stimulus.lags)
    winstep = round(Int, params.stimulus.samplerate/params.test.decode_sr)
    winlen = round(Int, params.test.winlen_s*params.stimulus.samplerate)
    target_index = round(Int, target_time * params.stimulus.samplerate)

    runsetup = @_ trialdf |>
        @where(__, :windowing .== "target") |>
        groupby(__, [:encoding, :source])

    result = combine(runsetup) do stimdf
        stimrow = only(eachrow(stimdf))
        train_type = stimrow.source == stimrow.target_source ? "athit-target" : "athit-other"
        source = @_ filter(string(_) == stimrow.source, sources) |> only
        encoding = encoding_map[stimrow.encoding]
        stim, = load_stimulus(source, sound_index, encoding, params.stimulus.samplerate,
            meta)

        maxlen = min(size(trialdata, 1), size(stim, 1))

        x = (view(trialdata, 1:maxlen, :) .- x_scores.μ') ./ x_scores.σ'
        y_μ, y_σ = stimrow[[:datamean, :datastd]]
        y = vec((view(stim, 1:maxlen, :) .- y_μ') ./ y_σ')

        modelrow = modelgroups[(
            source = string(source),
            encoding = stimrow.encoding,
            cross_fold = fold,
            train_type = train_type
        )] |> eachrow |> only
        ŷ = vec(modelrow.model(x'))

        function scoreat(offset)
            start = clamp(1+offset+target_index, 1, maxlen)
            stop = clamp(winlen+offset+target_index, 1, maxlen)

            if stop <= start
                Empty(DataFrame)
            else
                y_ = view(y, start:stop)
                ŷ_ = view(ŷ, start:stop)

                DataFrame(
                    score = cor(y_, ŷ_),
                    time = offset/params.stimulus.samplerate,
                    sid = sid,
                    trial = trial,
                    sound_index = sound_index,
                    fold = fold,
                    train_type = train_type,
                )
            end
        end

        start = round(Int, -3*params.stimulus.samplerate)
        stop = round(Int, 3*params.stimulus.samplerate)
        steps = start:winstep:stop
        # foldl(append!!, Map(scoreat), steps, init = Empty(DataFrame))
        foldxt(append!!, Map(scoreat), steps, init = Empty(DataFrame))
    end
    next!(progress)

    result
end
ProgressMeter.finish!(progress)

prefix = joinpath(processed_datadir("analyses", "decode-timeline"), "testing")
GermanTrack.@save_cache prefix timelines
