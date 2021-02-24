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
groups = @_ groupby(stimulidf, [:sid, :trial]) |>
    repeatby(__, :train_condition => unique(stimulidf.condition))

modelgroups = groupby(models_, [:condition, :source, :encoding, :cross_fold, :train_type])

p = Progress(ngroups(groups))
timelines = combine(groups) do trialdf
    sid, trial, sound_index, target_time, fold, condition =
        trialdf[:, [:sid, :trial, :sound_index, :target_time, :fold, :condition]] |>
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
        source = @_ filter(string(_) == stimrow.source, params.stimulus.sources) |> only
        encoding = params.stimulus.encodings[stimrow.encoding]
        stim, = load_stimulus(source, sound_index, encoding, params.stimulus.samplerate,
            meta)

        maxlen = min(size(trialdata, 1), size(stim, 1))

        x = (view(trialdata, 1:maxlen, :) .- x_scores.μ') ./ x_scores.σ'
        y_μ, y_σ = stimrow[[:datamean, :datastd]]
        y = vec((view(stim, 1:maxlen, :) .- y_μ') ./ y_σ')

        modelkey = (
            condition = stimrow.train_condition,
            source = string(source),
            encoding = stimrow.encoding,
            cross_fold = fold,
            train_type = train_type
        )
        haskey(modelgroups, modelkey) || return DataFrame()

        modelrow = modelgroups[modelkey] |> eachrow |> only
        ŷ = vec(modelrow.model(x'))

        function scoreat(offset)
            vstart = clamp(1+offset+target_index, 1, maxlen)
            vstop = clamp(winlen+offset+target_index, 1, maxlen)

            if vstop <= vstart
                Empty(DataFrame)
            else
                y_ = view(y, vstart:vstop)
                ŷ_ = view(ŷ, vstart:vstop)

                DataFrame(
                    score = cor(y_, ŷ_),
                    time = offset/params.stimulus.samplerate,
                    sid = sid,
                    trial = trial,
                    condition = condition,
                    sound_index = sound_index,
                    fold = fold,
                    train_type = train_type,
                )
            end
        end

        start = round(Int, -3*params.stimulus.samplerate)
        stop = round(Int, 3*params.stimulus.samplerate)
        steps = start:winstep:stop
        foldl(append!!, Map(scoreat), steps, init = Empty(DataFrame))
    end

    next!(p)
    result
end
ProgressMeter.finish!(p)

# Save the results
# -----------------------------------------------------------------

prefix = joinpath(processed_datadir("analyses", "decode-timeline"), "testing")
GermanTrack.@save_cache prefix timelines
