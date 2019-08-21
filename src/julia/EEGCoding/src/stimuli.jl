export encode_stimulus, RMSEnvelope, ASEnvelope, ASBins, TargetSuprisal, JointEncoding

abstract type StimEncoding
end

encode_stimulus(stim,tofs,target_time;method=RMSEnvelope()) =
    encode_stimulus(stim,tofs,target_time,method)

struct RMSEnvelope <: StimEncoding end
Base.string(::RMSEnvelope) = "rms_envelope"

function encode_stimulus(stim,tofs,_,::RMSEnvelope)
    N = round(Int,size(stim,1)/samplerate(stim)*tofs)
    result = zeros(N)
    window_size = 1.5/tofs
    toindex(t) = clamp(round(Int,t*samplerate(stim)),1,size(stim,1))

    for i in 1:N
        t = i/tofs
        from = toindex(t-window_size)
        to = toindex(t+window_size)
        result[i] = mean(x^2 for x in view(stim.data,from:to,:))
    end

    result
end

struct ASEnvelope <: StimEncoding end
Base.string(::ASEnvelope) = "audiospect_envelope"

function encode_stimulus(stim,tofs,_,::ASEnvelope)
    @assert size(stim,2) == 1

    spect_fs = CorticalSpectralTemporalResponses.fixed_fs
    resampled = Filters.resample(vec(stim),spect_fs/samplerate(stim))
    spect = filt(audiospect,SampleBuf(resampled,spect_fs),false)
    envelope = vec(sum(spect,dims=2))
    Filters.resample(envelope,ustrip(tofs*Δt(spect)))
end

struct ASBins <: StimEncoding
    bounds::Vector{Float64}
end
Base.string(bin::ASBins) = string("freqbins_",join(bin.bounds,"-"))

function encode_stimulus(stim,tofs,_,method::ASBins)
    @assert size(stim,2) == 1

    spect_fs = CorticalSpectralTemporalResponses.fixed_fs
    resampled = Filters.resample(vec(stim),spect_fs/samplerate(stim))
    spect = filt(audiospect,SampleBuf(resampled,spect_fs),false)
    f = frequencies(spect)
    bounds = zip([0.0Hz;method.bounds.*Hz],[method.bounds.*Hz;last(f)])
    bins = map(bounds) do (from,to)
        envelope = vec(sum(view(spect,:,from .< f .< to),dims=2))
        Filters.resample(envelope,ustrip(tofs*Δt(spect)))
    end
    hcat(bins)
end

struct TargetSuprisal <: StimEncoding

end Base.string(bin::TargetSuprisal) = "tpitch"

function encode_stimulus(stim,tofs,target_time,method::TargetSuprisal)
    len = ceil(Int,size(stim,1) * tofs/samplerate(stim))
    result = zeros(len)
    if target_time > 0
        target_frame = floor(Int,target_time * tofs)
        result[target_frame] = 1.0
    end

    result
end

struct JointEncoding <: StimEncoding
    children::Vector{StimEncoding}
end
JointEncoding(xs...) = JointEncoding(collect(xs))
Base.string(x::JointEncoding) = join(map(string,x.children),"_")

function encode_stimulus(stim,tofs,target_time,method::JointEncoding)
    encodings = map(method.children) do child
        encode_stimulus(stim,tofs,target_time,child)
    end

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
