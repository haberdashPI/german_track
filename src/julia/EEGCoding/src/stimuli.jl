using SignalOperators, CSV, AxisArrays
export RMSEnvelope, ASEnvelope, ASBins, PitchEncoding,
    PitchSurpriseEncoding, Stimulus, DiffEncoding, WeightedEncoding

struct Stimulus{A}
    data::A
    framerate::Float64
    file::Union{String,Nothing}
end
Stimulus(data,framerate,file) = Stimulus(data,Float64(framerate),file)

abstract type StimEncoding <: Encoding
end

"""
    encode(stim::Stimulus, tofs, [encoding])

Encode a stimulus so that we can use EEG encoding/decoding. Th default encoding
is `RMSEnvelope`, but there are a variety of other options.
"""
encode(x::Stimulus,tofs) = encode(x,tofs,RMSEnvelope())

"""
    RMSEnvelope()

Specifies a root-mean-squared envelope encoding. For each window of the stimulus
(1.5 seconds in length) find the RMS amplitude and return this series of amplitudes.
"""
struct RMSEnvelope <: StimEncoding end
Base.string(::RMSEnvelope) = "rms_envelope"

function encode(stim::Stimulus,tofs,::RMSEnvelope)
    N = round(Int,size(stim.data,1)/stim.framerate*tofs)
    result = zeros(N)
    window_size = 1.5/tofs
    toindex(t) = clamp(round(Int,t*stim.framerate),1,size(stim.data,1))

    for i in 1:N
        t = i/tofs
        from = toindex(t-window_size)
        to = toindex(t+window_size)
        result[i] = mean(x^2 for x in view(stim.data,from:to,:))
    end

    result
end

"""
    ASEnvelope

Compute an envelope using the auditory spectrogram. Roughly equivalent to computing
power in log-frequency bins and summing across all bins.
"""
struct ASEnvelope <: StimEncoding end
Base.string(::ASEnvelope) = "audiospect_envelope"

function encode(stim::Stimulus,tofs,::ASEnvelope)
    @assert size(stim.data,2) == 1

    @assert size(stim.data, 2) == 1
    resampled = DSP.resample(stim.data |> vec,
        CorticalSpectralTemporalResponses.fixed_fs / stim.framerate)
    spect = filt(audiospect,resampled,false,fs=8000)
    envelope = vec(sum(spect,dims=2))
    Filters.resample(envelope,ustrip(tofs*Δt(spect)))
end

"""
    ASBins(bounds)

Compute power within multiple bands of frequency, returning a matrix of time x bin.
The `bounds` should specific the frequency ranges to cut the bins at.
"""
struct ASBins <: StimEncoding
    bounds::Vector{Float64}
end
Base.string(bin::ASBins) = string("freqbins_",join(bin.bounds,"-"))

function encode(stim::Stimulus,tofs,method::ASBins)
    @assert size(stim.data,2) == 1

    spect_fs = CorticalSpectralTemporalResponses.fixed_fs
    resampled = Filters.resample(vec(stim.data),spect_fs/stim.framerate)
    spect = filt(audiospect,signal(resampled,spect_fs),false)
    f = frequencies(spect)
    bounds = zip([0.0Hz;method.bounds.*Hz],[method.bounds.*Hz;last(f)])
    bins = map(bounds) do (from,to)
        envelope = vec(sum(view(spect,:,from .< f .< to),dims=2))
        Filters.resample(envelope,ustrip(tofs*Δt(spect)))
    end
    hcat(bins)
end

# The JointEncoding shares the same definition as that for EEG data
# It simply concatenates representations along the columns.
function encode(stim::Stimulus,tofs,method::JointEncoding)
    encodings = map(x -> encode(stim,tofs,x),method.children)

    # like reduce(hcat,x) but pad the values with zero
    len = maximum(x -> size(x,1),encodings)
    width = sum(x -> size(x,2),encodings)
    result = zeros(len,width)
    col = 0
    for enc in encodings
        result[1:size(enc,1),(1:size(enc,2)) .+ col] = enc
        col += size(enc,2)
    end

    result
end

"""
    WeightedEncoding(weights, child)

Weight the features of `chidl` by `weights`. Changes the emphasis of a feature in the final
representation.
"""
struct WeightedEncoding{T} <: StimEncoding
    weights::Vector{Float64}
    child::T
end
Base.string(weighted::WeightedEncoding) =
    string("weight_",join(round.(weighted.weights,digits=1),"-"),
        string(weighted.child))
function encode(stim::Stimulus,tofs,method::WeightedEncoding)
    enc = encode(stim,tofs,method.child)
    enc .* method.weights'
end

"""
    DiffEncoding(child)

For each feature in `child`, compute `abs.(diff(x))` across time.
"""
struct DiffEncoding{T} <: StimEncoding
    child::T
end
Base.string(diff::DiffEncoding) = string("diff_",string(diff.child))
function encode(stim::Stimulus,tofs,method::DiffEncoding)
    enc = encode(stim,tofs,method.child)
    abs.(diff(enc,dims=1))
end

"""
    PitchEncoding()

Use a pre-computed pitch encoding of the stimulus, uses an auxilarly file ending in
`.f0.csv` to find the pitches.
"""
struct PitchEncoding <: StimEncoding
end
Base.string(::PitchEncoding) = "pitch"

load_pitch(file::Nothing) = nothing
function load_pitch(file)
    pitchfile = @_ replace(abspath(file), r"\.wav$" => ".f0.csv") |>
        replace(__, "mixture_component_channels" => "mixture_component_pitches")
    DataFrame(CSV.File(pitchfile))
end

function pitch_resample_helper(x,tofs,pitches)
    delta = pitches.time[2] - pitches.time[1]
    @assert all(x -> isapprox(delta,x),diff(pitches.time))
    if !isapprox(tofs*delta,1.0,atol=1e-4)
        DSP.resample(x,rationalize(tofs*delta))
    else
        x
    end
end

function encode(stim::Stimulus,file::String,tofs,method::PitchEncoding)::Array{Float64}
    pitches = load_pitch(stim.file)
    isnothing(pitches) ? Array{Float64}(undef,0,0) :
        pitch_resample_helper(pitches.frequency,tofs,pitches)
end

"""
    PitchSurpriseEncoding()

Use a pre-computed pitch encoding of the stimulus to compute pitch-surprisal, uses an
auxilarly file ending in `.f0.csv` to find the pitches.
"""
struct PitchSurpriseEncoding <: StimEncoding
end
Base.string(::PitchSurpriseEncoding) = "pitchsur"

function encode(stim::Stimulus,tofs,method::PitchSurpriseEncoding)::Array{Float64}
    pitches = load_pitch(stim.file)
    if isnothing(pitches)
        Array{Float64}(undef,0,0)
    else
        clean_nan(p,c) = iszero(p) ? zero(c) : c
        pitches.confidence = clean_nan.(pitches.frequency,pitches.confidence)

        surprisal = [0;abs.(diff(pitches.frequency)) .*
            @views(pitches.confidence[2:end])]
        pitch_resample_helper(surprisal,tofs,pitches)
    end
end
