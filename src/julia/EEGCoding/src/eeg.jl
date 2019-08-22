export EEGData, eegtrial, select_bounds, all_indices, no_indices, resample!,
    RawEncoding, FilteredPower
using DSP

Base.@kwdef struct EEGData
    label::Vector{String}
    fs::Int
    data::Vector{Matrix{Float64}}
end

SampledSignals.samplerate(x::EEGData) = x.fs
function SampledSignals.samplerate(x::MxArray)
    fs = mat"$x.fsample"
    fs
end

function eegtrial(x::MxArray,i)
    mat"response = $x.trial{$i};"
    response = get_mvariable(:response)
    response
end

function eegtrial(x::EEGData,i)
    x.data[i]
end

resample!(eeg::EEGData,::Nothing) = eeg
function resample!(eeg::EEGData,sr)
    @info "Resample EEG to $sr Hz."
    ratio = sr / samplerate(eeg)
    for i in eachindex(eeg.data)
        old = eeg.data[i]

        # first channel
        start = resample(view(old,1,:),ratio)
        eeg.data[i] = similar(old,size(old,1),length(start))
        eeg.data[i][1,:] = start

        # rest of the channels
        for chan in 2:size(old,1)
            eeg.data[i][chan,:] = DSP.resample(old[chan,:],ratio)
        end
    end
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

function select_bounds(x::MxArray,::AllIndices,min_len,fs,dim)
    if dim == 1
        mat"x = $x(1:min(end,$min_len),:);"
    elseif dim == 2
        mat"x = $x(:,1:min(end,$min_len));"
    end
    get_mvariable(:x)
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

function select_bounds(x::MxArray,(start,stop)::Tuple,min,fs,dim)
    start,stop = toindex.((start,stop),min,fs)
    if dim == 1
        mat"x = $x($start:$stop,:);"
    elseif dim == 2
        mat"x = $x(:,$start:$stop);"
    else
        error("Unspported dimension $dim.")
    end

    get_mvariable(:x)
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

function select_bounds(x::MxArray,bounds::AbstractArray{<:Tuple},min,fs,dim)
    mat"indices = [];"
    for (start,stop) in bounds
        start,stop = toindex.((start,stop),min,fs)
        mat"indices = [indices $start:$stop];"
    end
    if dim == 1
        mat"x = $x(indices,:);"
    elseif dim == 2
        mat"x = $x(:,indices);"
    else
        error("Unspported dimension $dim.")
    end

    get_mvariable(:x)
end

struct RawEncoding <: Encoding
end
Base.string(::RawEncoding) = "raw"
function encode(x::EEGData,tofs,::RawEncoding)
    resample!(x,tofs)
end

abstract type EEGEncoding <: Encoding
end

struct FilteredPower{D} <: EEGEncoding
    name::String
    from::Float64
    to::Float64
    design::D
end
FilteredPower(name,from,to,order=5,filter=Butterworth(order)) =
    FilteredPower(name,from,to,filter)
Base.string(x::FilteredPower) = x.name
function encode(x::EEGData,tofs,filter::FilteredPower)
    bandpass = digitalfilter(Bandpass(from,to,samplerate(x)),filter.design)
    power = similar(x.data)
    for trial in 1:length(x.data)
        power[trial] = similar(x.data[trial])
        for i in 1:size(x.data,1)
            power[trial][i,:] .=
                abs.(DSP.Util.hilbert(filt(bandpass,view(x.data,i,:))))
        end
    end
    @info "Resample $(filter.name) power to $resample Hz."
    resample!(EEGData(string.(x.label,"_",filter.name),x.fs,power),tofs)
end

function encode(x::EEGData,tofs,joint::JointEncoding)
    encodings = map(joint.children) do method
        encode(x,tofs,method)
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
