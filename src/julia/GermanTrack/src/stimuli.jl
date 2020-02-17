using EEGCoding
using SignalOperators
const encodings = Dict{Any,Array{Float64}}()
export SpeakerStimMethod, joint_source, male_source, fem1_source, fem2_source,
    other, mixed_sources, fem_mix_sources

abstract type StimMethod
end

abstract type AbstractSource
end
fortraining(x::AbstractSource) = x

function encode_cache(body,key,stim_num)
    if key ∈ keys(encodings)
        encodings[key], stim_num
    else
        result = body()
        encodings[key] = result
        result, stim_num
    end
end
clear_stim_cache!() = empty!(encodings)

struct SingleSource <: AbstractSource
    name::String
    index::Int
end
Base.string(x::SingleSource) = x.name
male_source = SingleSource("male",1)
fem1_source = SingleSource("fem1",2)
fem2_source = SingleSource("fem2",3)
function load_stimulus(source::SingleSource,stim_i,stim_method,events,tofs,stim_info)
    stim_num = events.sound_index[stim_i]
    target_time = events.target_source[stim_i] == source.index ?
        events.target_time[stim_i] : nothing
    load_single_speaker(tofs,stim_num,source.index,target_time,
        stim_method.encoding)
end

function load_single_speaker(tofs,stim_num,source_i,target_time,encoding)
    encode_cache((:speaker,tofs,stim_num,source_i,encoding),stim_num) do
        file = joinpath(stimulus_dir(),"mixtures","testing","mixture_components",
            @sprintf("trial_%02d_%1d.wav",stim_num,source_i))
        x,fs = load(file)
        if size(x,2) > 1
            x = sum(x,dims=2)
        end
        encode(Stimulus(x,fs,file,target_time),tofs,encoding)
    end
end

struct JointSource <: AbstractSource
end
joint_source = JointSource()
Base.string(::JointSource) = "joint"

function load_stimulus(source::JointSource,stim_i,stim_method,events,
    tofs,stim_info)

    stim_num = events.sound_index[stim_i]
    target_time = events.target_time[stim_i]
    load_joint_stimulus(events,tofs,stim_num,target_time,
        stim_method.encoding)
end

function load_joint_stimulus(events,tofs,stim_num,target_time,encoding)

    encode_cache((:joint,tofs,stim_num,encoding),stim_num) do
        fs = 0
        stimdir = joinpath(stimulus_dir(),"mixtures","testing",
            "mixture_components")
        sources = (joinpath(stimdir,@sprintf("trial_%02d_%1d.wav",stim_num,j))
            for j in 1:3)
        mapreduce(hcat,sources) do file
            x,fs = load(file)
            if size(x,2) > 1
                x = sum(x,dims=2)
            end
            encode(Stimulus(x,fs,file,target_time),tofs,encoding)
        end
    end
end

struct OtherSource{S} <: AbstractSource
    source::S
end
other(x::AbstractSource) = OtherSource(x)
other(x::OtherSource) = error("Already 'othered'")
fortraining(x::OtherSource) = x.source
Base.string(x::OtherSource) = string("other_",string(x.source))

function load_stimulus(::OtherSource{JointSource},stim_i,stim_method,events,
    tofs,info)

    stim_num = events.sound_index[stim_i]
    selected = rand(filter(@λ(_ != stim_num),1:50))

    target_time = events.target_time[stim_i]
    result, real_stim_num =
        load_joint_stimulus(events,tofs,selected,target_time,
            stim_method.encoding)
    result, stim_num
end

function load_stimulus(other::OtherSource{SingleSource},stim_i,stim_method,
    events,tofs,info)

    stim_num = events.sound_index[stim_i]
    stimuli = info["test_block_cfg"]["trial_sentences"]
    sentence_num = stimuli[stim_num][other.source.index]
    selected = rand(filter(i -> stimuli[i][other.source.index] != sentence_num,
        1:length(stimuli)))

    target_time = events.target_source[stim_i] == other.source.index ?
        events.target_time[stim_i] : nothing
    result, real_stim_num =
        load_single_speaker(tofs,selected,other.source.index,target_time,
            stim_method.encoding)
    result, stim_num
end

struct MixedSources <: AbstractSource
    indices::Vector{Int}
    name::String
end
Base.string(x::MixedSources) = x.name
mixed_sources = MixedSources(1:3,"all")
fem_mix_sources = MixedSources(2:3,"fem1+fem2")

function load_stimulus(mixed::MixedSources,stim_i,stim_method,events,tofs,info)
    stim_num = events.sound_index[stim_i]
    key = (:mixed,mixed.indices,stim_i,tofs,stim_num,stim_method.encoding)
    encode_cache(key,stim_num) do
        filenames = map(mixed.indices) do source_i
            joinpath(stimulus_dir(),"mixtures","testing",
                "mixture_components",
                @sprintf("trial_%02d_%1d.wav",stim_num,source_i))
        end
        mixture, fr = Mix(filenames...) |> ToChannels(1) |> sink

        target_time = events.target_source[stim_i] ∈ mixed.indices ?
            events.target_time[stim_i] : nothing
        encode(Stimulus(mixture,fr,nothing,target_time),
            tofs,stim_method.encoding)
    end
end

const all_sources =
    [male_source, fem1_source, fem2_source, joint_source, other(joint_source),
     mixed_sources]

Base.@kwdef struct SpeakerStimMethod <: StimMethod
    sources::Vector = all_sources
    encoding::EEGCoding.Encoding
end
label(x::SpeakerStimMethod) = "speakers_"*string(x.encoding)
sources(x::SpeakerStimMethod) = unique!(fortraining.(x.sources)), x.sources
