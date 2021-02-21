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

nfolds = 5

# STEPS: maybe we should consider cross validating across stimulus type
# rather than subject id?

# Load subject data
# -----------------------------------------------------------------

samplerate = 32

prefix = joinpath(cache_dir("eeg", "decoding"), "eeg-training-data")
GermanTrack.@use_cache prefix (subjects, :jld) begin
    @info "Resampling EEG data, this may take a while (this step will be cached to avoid repeating it)"
    # eeg_encoding = FFTFilteredPower("freqbins", Float32[1, 3, 7, 15, 30, 100])
    eeg_encoding = JointEncoding(
        RawEncoding(),
        FilteredPower("delta", 1,  3),
        FilteredPower("theta", 3,  7),
        FilteredPower("alpha", 7,  15),
        FilteredPower("beta",  15, 30),
        FilteredPower("gamma", 30, 100),
    )
    # eeg_encoding = RawEncoding()

    subjects, = load_all_subjects(processed_datadir("eeg"), "h5",
        encoding = eeg_encoding, framerate = samplerate)
end

# things that are duplicated in `train_decode`: should be defined in a common file
# and written out to a TOML parameter setup

max_lag = 3
nlags = round(Int,samplerate*max_lag)
lags = -(nlags-1):1:0

sources = [
    male_source,
    fem1_source,
    fem2_source,
    # male_fem1_sources,
    # male_fem2_sources,
    # fem1_fem2_sources
]

encoding_map = Dict("pitch" => PitchSurpriseEncoding(), "envelope" => ASEnvelope())

decode_prefix = joinpath(processed_datadir("analyses", "decode"), "train")
GermanTrack.@load_cache decode_prefix (models_, :bson) stimulidf x_scores

meta = GermanTrack.load_stimulus_metadata()

# timeline testing
# -----------------------------------------------------------------

decode_sr = 1 / (round(Int, 0.1samplerate) / samplerate)
winlen_s = 1.0

sid_trial = mapreduce(x -> x[1] .=> eachindex(x[2].eeg.data), vcat, pairs(subjects))
groups = groupby(stimulidf, [:sid, :trial, :condition])
progress = Progress(length(groups), desc = "Evaluating decoder over timeline...")

modelgroups = groupby(models_, [:source, :encoding, :cross_fold, :train_type])

timelines = combine(groups) do trialdf
    sid, trial, sound_index, target_time, fold =
        trialdf[:, [:sid, :trial, :sound_index, :target_time, :fold]] |>
        eachrow |> unique |> only

    trialdata = withlags(subjects[sid].eeg[trial]', lags)
    winstep = round(Int, samplerate/decode_sr)
    winlen = round(Int, winlen_s*samplerate)
    target_index = round(Int, target_time * samplerate)

    runsetup = @_ trialdf |>
        @where(__, :windowing .== "target") |>
        groupby(__, [:encoding, :source])

    result = combine(runsetup) do stimdf
        stimrow = only(eachrow(stimdf))
        train_type = stimrow.source == stimrow.target_source ? "athit-target" : "athit-other"
        source = @_ filter(string(_) == stimrow.source, sources) |> only
        encoding = encoding_map[stimrow.encoding]
        stim, = load_stimulus(source, sound_index, encoding, samplerate, meta)

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
                    time = offset/samplerate,
                    sid = sid,
                    trial = trial,
                    sound_index = sound_index,
                    fold = fold,
                    train_type = train_type,
                )
            end
        end

        steps = round(Int, -3*samplerate):winstep:round(Int, 3*samplerate)
        # foldl(append!!, Map(scoreat), steps, init = Empty(DataFrame))
        foldxt(append!!, Map(scoreat), steps, init = Empty(DataFrame))
    end
    next!(progress)

    result
end
ProgressMeter.finish!(progress)

prefix = joinpath(processed_datadir("analyses", "decode-timeline"), "testing")
GermanTrack.@save_cache prefix timelines
