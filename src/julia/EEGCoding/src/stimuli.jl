using SampledSignals, CSV
export RMSEnvelope, ASEnvelope, ASBins, PitchEncoding,
    PitchSurpriseEncoding, Stimulus, DiffEncoding

struct Stimulus
    data::SampleBuf
    file::Union{String,Nothing}
    target_time::Union{Float64,Nothing}
end
function Stimulus(data::Array,fs::Number,file::Union{String,Nothing},
    target_time::Union{Float64,Nothing}=nothing)

    Stimulus(SampleBuf(data,fs),file,target_time)
end

abstract type StimEncoding <: Encoding
end

encode(x::Stimulus,tofs) = encode(x,tofs,RMSEnvelope())

struct RMSEnvelope <: StimEncoding end
Base.string(::RMSEnvelope) = "rms_envelope"

function encode(stim::Stimulus,tofs,::RMSEnvelope)
    N = round(Int,size(stim.data,1)/samplerate(stim.data)*tofs)
    result = zeros(N)
    window_size = 1.5/tofs
    toindex(t) = clamp(round(Int,t*samplerate(stim.data)),1,size(stim.data,1))

    for i in 1:N
        t = i/tofs
        from = toindex(t-window_size)
        to = toindex(t+window_size)
        result[i] = mean(x^2 for x in view(stim.data.data,from:to,:))
    end

    result
end

struct ASEnvelope <: StimEncoding end
Base.string(::ASEnvelope) = "audiospect_envelope"

function encode(stim::Stimulus,tofs,::ASEnvelope)
    @assert size(stim.data,2) == 1

    spect_fs = CorticalSpectralTemporalResponses.fixed_fs
    resampled = Filters.resample(vec(stim.data),spect_fs/samplerate(stim.data))
    spect = filt(audiospect,SampleBuf(resampled,spect_fs),false)
    envelope = vec(sum(spect,dims=2))
    Filters.resample(envelope,ustrip(tofs*Δt(spect)))
end

struct ASBins <: StimEncoding
    bounds::Vector{Float64}
end
Base.string(bin::ASBins) = string("freqbins_",join(bin.bounds,"-"))

function encode(stim::Stimulus,tofs,method::ASBins)
    @assert size(stim.data,2) == 1

    spect_fs = CorticalSpectralTemporalResponses.fixed_fs
    resampled = Filters.resample(vec(stim.data),spect_fs/samplerate(stim.data))
    spect = filt(audiospect,SampleBuf(resampled,spect_fs),false)
    f = frequencies(spect)
    bounds = zip([0.0Hz;method.bounds.*Hz],[method.bounds.*Hz;last(f)])
    bins = map(bounds) do (from,to)
        envelope = vec(sum(view(spect,:,from .< f .< to),dims=2))
        Filters.resample(envelope,ustrip(tofs*Δt(spect)))
    end
    hcat(bins)
end

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

struct DiffEncoding{T}
    child::T
end
Base.string(diff::DiffEncoding) = string("diff_",string(dff.child))
function encode(stim::Stimulus,tofs,method::DiffEncoding)
    enc = encode(stim,tofs,method.child)
    abs.(diff(enc,dims=1))
end

struct PitchEncoding <: StimEncoding
end
Base.string(::PitchEncoding) = "pitch"

load_pitch(file::Nothing) = nothing
function load_pitch(file)
    pitchfile = replace(file,r"\.wav$" => ".f0.csv")
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
