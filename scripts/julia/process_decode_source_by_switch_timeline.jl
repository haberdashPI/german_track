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

decode_prefix = joinpath(processed_datadir("analyses", "decode"), "train")
GermanTrack.@load_cache decode_prefix stimulidf x_scores (models_, :bson)

subject_prefix = joinpath(mkpath(processed_datadir("analyses", "decode-data")), "freqbinpower-sr$(params.stimulus.samplerate)")
GermanTrack.@load_cache subject_prefix (subjects, :bson)

# things that are duplicated in `train_decode`: should be defined in a common file
# and written out to a TOML parameter setup

models = @_ models_ |> rename!(__, :cross_fold => :fold, :source => :trained_source) |>
    insertcols!(__, :lagcut => 0)

meta = GermanTrack.load_stimulus_metadata()

# decode_prefix = joinpath(processed_datadir("analyses", "decode-varlag"), "train")
# GermanTrack.@load_cache decode_prefix (models_, :bson)

# append!(models, rename!(models_, :cross_fold => :fold, :source => :trained_source))

nfeatures = floor(Int, size(first(subjects)[2].eeg[1], 1))

# timeline testing
# -----------------------------------------------------------------

groups = @_ stimulidf |>
    @where(__, :windowing .== "random1") |>
    @where(__, (:condition .== "global") .|
               ((:condition .== "object") .& (.!contains.(:source, r"\(ch [12]\)"))) .|
               ((:condition .== "spatial") .& (contains.(:source, r"\(ch [12]\)")))) |>
    groupby(__, [:sid, :trial])

cutlags(x, nfeatures, ncut) = (ncut*nfeatures+1):size(x, 2)

sourcename(str) = match(r"\w+", str).match
p = Progress(ngroups(groups))
timelines = combine(groups) do trialdf
    sid, trial, sound_index, target_time, fold, condition =
        trialdf[:, [:sid, :trial, :sound_index, :target_time, :fold, :condition]] |>
        unique |> eachrow |> only

    switches = meta.switch_regions[sound_index]

    lagcuts = levels(models.lagcut)
    runsetup = @_ copy(trialdf) |>
        @where(__, (:hittype .∈ Ref(["miss", "hit"]))) |>
        # @transform(__, lagcut = 0) |>
        @repeatby(__,
            switch_index = 1:length(switches),
            trained_source = levels(:source),
            lagcut = levels(models.lagcut)
        ) |>
        @where(__, ifelse.(contains.(:trained_source, r"\(ch [12]\)"),
            sourcename.(:source) .== sourcename.(:trained_source),
            .!contains.(:source, r"\(ch [12]\)"))) |>
        innerjoin(__, models,
            on = [:condition, :trained_source, :encoding, :fold, :lagcut]) |>
        combine(identity, __)

    isempty(runsetup) && return DataFrame()

    trialdata = withlags(subjects[sid].eeg[trial]', params.stimulus.lags)
    winstep = round(Int, params.stimulus.samplerate/params.test.decode_sr)
    winlen = round(Int, params.test.winlen_s*params.stimulus.samplerate)
    target_index = round(Int, target_time * params.stimulus.samplerate)

    result = combine(groupby(runsetup, [:encoding, :trained_source, :source, :switch_index, :lagcut])) do stimdf
        stimrow = only(eachrow(stimdf))
        source = @_ filter(string(_) == stimrow.source, params.stimulus.sources) |> only
        encoding = params.stimulus.encodings[stimrow.encoding]
        stim, = load_stimulus(source, sound_index, encoding, params.stimulus.samplerate,
            meta)

        maxlen = min(size(trialdata, 1), size(stim, 1),
            round(Int, params.train.trial_time_limit * params.stimulus.samplerate))

        lags = cutlags(trialdata, nfeatures, stimrow.lagcut)
        x = (view(trialdata, 1:maxlen, lags) .- view(x_scores.μ, lags)') ./
            view(x_scores.σ, lags)'
        y_μ, y_σ = stimrow[[:datamean, :datastd]]
        y = vec((view(stim, 1:maxlen, :) .- y_μ') ./ y_σ')

        ŷ = vec(gpu(stimrow.model)(gpu(x'))) |> cpu

        switch_time_index = round(Int, switches[stimrow.switch_index][2] *
            params.stimulus.samplerate)

        function scoreat(offset)
            vstart = clamp(1+offset+switch_time_index, 1, maxlen)
            vstop = clamp(winlen+offset+switch_time_index, 1, maxlen)

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
                    is_target_source = stimdf.is_target_source,
                    fold = fold,
                    lagcut = stimrow.lagcut,
                    switch_index = stimrow.switch_index,
                    hittype = stimrow.hittype,
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

# Save results
# -----------------------------------------------------------------

prefix = joinpath(processed_datadir("analyses", "decode-timeline-switch"), "testing")
GermanTrack.@save_cache prefix timelines

