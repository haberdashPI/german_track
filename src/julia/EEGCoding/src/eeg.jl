export EEGData, select_bounds, all_indices, no_indices, resample!,
    RawEncoding, FilteredPower, FFTFiltered
using DSP
using DataStructures

Base.@kwdef mutable struct EEGData
    label::Vector{String}
    fs::Int
    data::Vector{Matrix{Float64}}
end
Base.copy(x) = EEGData(copy(x.label),x.fs,copy(x.data))

SignalOperators.framerate(x::EEGData) = x.fs
Base.getindex(x::EEGData,i) = x.data[i]
SignalOperators.nchannels(x::EEGData) = size(x.data[1], 1)
maxframes(x::EEGData) = maximum(x -> size(x, 2), x.data)
trial_eltype(x::EEGData) = eltype(x.data[1])

resample!(eeg::EEGData,::Missing) = eeg
function resample!(eeg::EEGData,sr)
    ratio = sr / framerate(eeg)
    if ratio ≈ 1
        return eeg
    end

    # progress = Progress(length(eeg.data))
    Threads.@threads for i in eachindex(eeg.data)
        old = eeg.data[i]

        # first channel
        start = resample(view(old,1,:),ratio)
        eeg.data[i] = similar(old,size(old,1),length(start))
        eeg.data[i][1,:] = start

        # rest of the channels
        for chan in 2:size(old,1)
            eeg.data[i][chan,:] = DSP.resample(old[chan,:],ratio)
        end

        # next!(progress)
    end
    eeg.fs = sr

    eeg
end
################################################################################
# handle selecting various bounds of a signal

# denotes selection of all valid time indices
struct AllIndices end
const all_indices = AllIndices()
Base.getindex(x::AllIndices,i::Int) = x
Base.isempty(x::AllIndices) = false
(x::AllIndices)(row) = all_indices

struct NoIndices end
const no_indices = NoIndices()
Base.isempty(x::NoIndices) = true
(x::NoIndices)(row) = no_indices

toindex(x,min,fs) = clamp.(round.(Int,x.*fs),1,min)

function select_bounds(x::AbstractArray,::AllIndices,min_len,fs,dim)
    if dim == 1
        x[1:min(min_len,end),:]
    elseif dim == 2
        x[:,1:min(min_len,end)]
    end
end

function select_bounds(x::AbstractArray,(start,stop)::Tuple,min,fs,dim)
    start,stop = toindex.((start,stop),min,fs)
    if dim == 1
        x[start:stop,:]
    elseif dim ==2
        x[:,start:stop]
    else
        error("Unspported dimension $dim.")
    end
end

function select_bounds(x::AbstractArray,bounds::AbstractArray{<:Tuple},min,fs,dim)
    if dim == 1
        vcat(select_bounds.(Ref(x),bounds,min,fs,dim)...)
    elseif dim == 2
        hcat(select_bounds.(Ref(x),bounds,min,fs,dim)...)
    else
        error("Unspported dimension $dim.")
    end
end

struct RawEncoding <: Encoding
end
Base.string(::RawEncoding) = "raw"
function encode(x::EEGData,tofs,::RawEncoding)
    if !ismissing(tofs)
        # @info "Resample EEG to $tofs Hz."
    end
    resample!(x,tofs)
end

abstract type EEGEncoding <: Encoding
end

struct FFTFiltered{P} <: EEGEncoding
    bands::OrderedDict{String,Tuple{Float64,Float64}}
    plan::P
    buffer::Array{Float64,2}
end
Base.string(x::FFTFiltered) = join((keys(x.bands)...,"filtering"),"_")
FFTFiltered(pairs::Pair...;seconds,fs,nchannels) =
    FFTFiltered(OrderedDict(pairs...),seconds,fs,nchannels)
function FFTFiltered(bands::OrderedDict,seconds,fs,nch)
    n = floor(Int,seconds*fs)
    buffer = Array{Float64}(undef,nch,n)
    plan = plan_rfft(buffer,2,flags=FFTW.PATIENT,timelimit=4)

    if any(fs .<= (2 .* Iterators.flatten(values(bands))))
        error("Frequency bands exceede Nyquist frequency.")
    end

    FFTFiltered(bands,plan,buffer)
end
function encode(x::EEGData,tofs,filter::FFTFiltered)
    if framerate(x) != tofs
        # @info "Resample EEG to $tofs Hz."
    end
    x = resample!(x,tofs)
    labels = mapreduce(vcat,keys(filter.bands)) do band
        x.label .* "_" .* band .* "_filtering"
    end
    trials = map(x.data) do trial
        if size(trial,2) > size(filter.buffer,2)
            clip = size(trial,2) -  size(filter.buffer,2)
            @warn "Clipping $(clip/tofs) seconds from eeg."
            trial = view(trial,:,Base.axes(filter.buffer,2))
        end
        filter.buffer[:,Base.axes(trial,2)] .= trial
        filter.buffer[:,(Base.size(trial,2)+1):end] .= 0
        result = filter.plan * filter.buffer

        freqs = range(0,tofs/2,length=size(result,2))
        mapreduce(vcat,values(filter.bands)) do (from,to)
            filtered = copy(result)
            filtered[:,findall((freqs .< from) .| (to .< freqs))] .= 0
            filter.plan \ filtered
        end
    end
    EEGData(labels,tofs,trials)
end

struct FFTFilteredPower <: EEGEncoding
    name::String
    cuts::Vector{Float64}
end

function find_sample_step(fromfs, tofs)
    sample_step = round(Int, fromfs / tofs)
    err = (fromfs/cutdown - tofs)/tofs
    if err > 1.05 || err < 0.95
        @warn "Imperfect resampling, true resample rate of $(fromfs/cutdown)."
    end

    sample_step
end

function encode(x::EEGData, tofs, filter::FFTFilteredPower)
    binpower = similar(x.data)
    fromfs = framerate(x)

    step = find_sample_step(fromfs, tofs)
    n, m = nchannels(x), nextprod([2], maxframes(x))
    mid = m >> 1

    buffers = [Array{trial_eltype(x)}(undef, m, n) for _ in 1:Threads.nthreads()]
    ibuffer = [Array{complex(trial_eltype(x)),2}(undef, m, n) for _ in 1:Threads.nthreads()]
    plan = plan_fft!(buffer, 1, flags = FFTW.PATIENT, timelimit = 4)
    iplan = plat_ifft!(ibuffer, 1, flags = FFTW.PATIENT, timelimit = 4)

    nbins = length(filter.cuts) - 1
    cuts = round(Int, mid .* filter.cuts ./ (fromfs/2))

    splitdims = size(ibuffer)
    splits = [similar(x.data[trial], splitdims) for _ in 1:nbins]
    splitindices = Array{Array{Int}}(undef, nbins)
    for bin in 1:nbins
        splitindices[bin] = vcat(
            mid .- (cut[bin+1]:-1:cut[bin]) .+ 1,
            mid .+ (cut[bin]:cut[bin+1])
        )
    end

    Threads.@threads for trial in 1:length(x.data)
        tid = Threads.threadid()
        len = size(x.data[trial],2)
        result = similar(x.data, (len, n*nbins))

        buffer[tid][1:len, :] = x.data[trial]'
        buffer[tid][(len+1):end, :] .= 0
        freqs = buffer[tid] * plan

        freqs[1:mid, :] .*= im
        freqs[(mid+1):end, :] .*= -im

        binpower[trial] = Array{trial_eltype(x)(undef, n*nbins, len)}
        for bin in 1:nbins
            ibuffer[tid] .= 0
            ibuffer[tid][splitindices[bin]] = freqs[splitindices[bin]]
            result[:, (1:n) .+ (1-bin)*n] = abs.(ibuffer[tid] * iplan)[1:len, :]
        end

        for ch in 1:size(result,1)
            binpower[trial][ch,:] = DSP.resample(result[:,ch], 1//step)
        end
    end
    EEGData(x.label, tofs, binpower)
end

struct FilteredPower{D} <: EEGEncoding
    name::String
    from::Float64
    to::Float64
    design::D
end
FilteredPower(name,from,to;order=5,filter=Butterworth(order)) =
    FilteredPower(string(name),Float64(from),Float64(to),filter)
Base.string(x::FilteredPower) = x.name + "_power"
function encode(x::EEGData,tofs,filter::FilteredPower)
    bp = Bandpass(filter.from,filter.to,fs=framerate(x))
    bandpass = digitalfilter(bp,filter.design)
    power = similar(x.data)
    for trial in 1:length(x.data)
        power[trial] = similar(x.data[trial])
        for i in 1:size(x.data[trial],1)
            power[trial][i,:] .=
                abs.(DSP.Util.hilbert(filt(bandpass,view(x.data[trial],i,:))))
        end
    end
    # @info "Resample $(filter.name) power to $tofs Hz."
    resample!(EEGData(string.(x.label,"_",filter.name),x.fs,power),tofs)
end

function encode(x::EEGData,tofs,joint::JointEncoding)
    encodings = map(joint.children) do method
        encode(copy(x),tofs,method)
    end
    labels = mapreduce(x -> x.label,vcat,encodings)
    fs = first(encodings).fs
    data = similar(first(encodings).data)
    for trial in 1:length(data)
        data[trial] = mapreduce(vcat,encodings) do enc
            enc.data[trial]
        end
    end
    EEGData(labels,fs,data)
end
