# Setup
# =================================================================

using DrWatson; @quickactivate("german_track")
using EEGCoding, GermanTrack, DataFrames, StatsBase, Underscores, Transducers,
    BangBang, ProgressMeter, HDF5

traindata = joinpath(cache_dir(), "data") |> mkpath
trainfile = joinpath(traindir, "decode-train.h5")
if isfile(trainfile)
    x, y = h5open(trainfile, "r") do stream
        data["x"], data["y"]
    end
else
    eeg_encoding = FFTFilteredPower("freqbins", Float64[1, 3, 7, 15, 30, 100])

    sr = 32
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5",
        encoding = eeg_encoding, framerate = sr)
    meta = GermanTrack.load_stimulus_metadata()

    windows = @_ events |>
        filter(ishit(_) == "hit", __) |> eachrow |>
        foldl(push!!, init = Empty(Vector), Map(trial -> (
            start = max(1, round(Int, sr*meta.target_times[trial.sound_index])),
            len = max(1, round(Int, sr*meta.trial_lengths[trial.sound_index])),
            trialnum = trial.trial,
            trial[[:condition, :sid]]...
        )), __) |>
        DataFrame

    ntimes = windows.len |> maximum
    nsegments = @_ windows |> size(__, 1)
    nmcca = size(first(subjects)[2].eeg[1],1)
    nfreqbins = eeg_encoding.children |> length
    nfeatures = nmcca * nfreqbins
    nlags = round(Int,sr*2)

    x = Array{Float32}(undef, ntimes, nfeatures*nlags, nsegments)
    progress = Progress(size(windows, 1), desc = "Organizing EEG data...")
    Threads.@threads for (i, trial) in enumerate(eachrow(windows))
        start = trial.start
        stop = trial.start + trial.len - 1
        trialdata = withlags(subjects[trial.sid].eeg[trial.trialnum]', -(nlags-1):0)
        x[1:trial.len,:,i] = @view(trialdata[start:stop, :])
        next!(progress)
    end

    stim_encoding = JointEncoding(PitchSurpriseEncoding(), ASEnvelope())
    nenc = length(stim_encoding.children)
    y = Array{Float32}(undef, ntimes, nenc, nsegments)
    progress = Progress(size(windows, 1), desc = "Organizing EEG data...")
    Threads.@threads for (i, trial) in enumerate(eachrow(windows))
        source = trial.target_source == "male" ? male_source : fem1_source
        stim, stim_id = load_stimulus(source, trial, stim_encoding, tofs, meta)
        start = trial.start
        stop = min(size(stim,1), trial.start + start.len - 1)
        if stop >= start
            len = stop - start + 1
            y[1:len, :, i] = @view(stim[start:stop, :])
            y[(len+1):end, :, i] .= zero(eltype(y))
        else
            y[:, :, i] .= zero(eltype(y))
        end
        next!(progress)
    end

    h5open(trainfile, "w") do stream
        stream["x"] = x
        stream["y"] = y
    end
end







