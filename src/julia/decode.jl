# Setup
# =================================================================

using DrWatson; @quickactivate("german_track")
using EEGCoding, GermanTrack, DataFrames, StatsBase, Underscores, Transducers,
    BangBang, ProgressMeter, HDF5

# eeg_encoding = FFTFilteredPower("freqbins", Float64[1, 3, 7, 15, 30, 100])
# eeg_encoding = JointEncoding(
#     FilteredPower("delta", 1, 3),
#     FilteredPower("theta", 3, 7),
#     FilteredPower("alpha", 7, 15),
#     FilteredPower("beta", 15, 30),
#     FilteredPower("gamma", 30, 100),
# )
eeg_encoding = RawEncoding()

sr = 32
subjects, events = load_all_subjects(processed_datadir("eeg"), "h5",
    encoding = eeg_encoding, framerate = sr)
meta = GermanTrack.load_stimulus_metadata()

target_length = 1.0
max_lag = 2.0

target_samples = round(Int, sr*target_length)
windows = @_ events |>
    filter(ishit(_) == "hit", __) |> eachrow |>
    map(function(event)
        triallen     = size(subjects[event.sid].eeg[event.trial], 2)
        start        = clamp(round(Int, sr*meta.target_times[event.sound_index]), 1,
                            triallen)
        len          = clamp(target_samples, 1, triallen-start)
        (
            start    = start,
            len      = len,
            trialnum = event.trial,
            event[[:condition, :sid, :target_source, :sound_index]]...
        )
        end, __) |>
    DataFrame

nobs = sum(windows.len)
starts = vcat(1,1 .+ cumsum(windows.len))
nfeatures = size(first(subjects)[2].eeg[1],1)
nlags = round(Int,sr*max_lag)
x = Array{Float64}(undef, nobs, nfeatures*nlags)

progress = Progress(size(windows, 1), desc = "Organizing EEG data...")
Threads.@threads for (i, trial) in collect(enumerate(eachrow(windows)))
    start = trial.start
    stop = trial.start + trial.len - 1
    trialdata = withlags(subjects[trial.sid].eeg[trial.trialnum]', -(nlags-1):0)
    x[starts[i] : (starts[i+1]-1), :] = @view(trialdata[start:stop, :])
    next!(progress)
end

stim_encoding = JointEncoding(PitchSurpriseEncoding(), ASEnvelope())
nenc = length(stim_encoding.children)
y = Array{Float64}(undef, nobs, nenc)

progress = Progress(size(windows, 1), desc = "Organizing stimulus data...")
for (i, trial) in enumerate(eachrow(windows))
    source = trial.target_source == "male" ? male_source : fem1_source
    stim, stim_id = load_stimulus(source, trial, stim_encoding, sr, meta)
    start = trial.start
    stop = min(size(stim,1), trial.start + trial.len - 1)
    if stop >= start
        len = stop - start + 1
        y[starts[i] : (starts[i] + len - 1), :] = @view(stim[start:stop, :])
        y[(starts[i] + len) : (starts[i+1] - 1), :] .= zero(eltype(y))
    else
        y[:, :, i] .= zero(eltype(y))
    end
    next!(progress)
end

decode(x,y) = fit(LassoPath, x, y)
decoders = foldxt(push!!, init = Empty(Vector), Map(i -> decode(x,y[:,i])), axes(y, 2))
