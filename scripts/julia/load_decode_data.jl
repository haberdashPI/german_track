using DrWatson
@quickactivate("german_track")
println("Julia version: $VERSION.")

using EEGCoding, GermanTrack, DataFrames, PooledArrays
include(joinpath(scriptsdir(), "julia", "setup_decode_params.jl"))

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
    encoding = eeg_encoding, framerate = params.stimulus.samplerate)

prefix = joinpath(mkpath(processed_datadir("analyses", "decode-data")), "freqbinpower-sr$(params.stimulus.samplerate)")
GermanTrack.@save_cache prefix (subjects, :bson)
