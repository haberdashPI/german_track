using EEGCoding
const encodings = Dict{Any,Array{Float64}}()
export SpeakerStimMethod, ChannelStimMethod

abstract type StimMethod
end

Base.@kwdef struct SpeakerStimMethod <: StimMethod
    encoding::EEGCoding.Encoding
end
label(x::SpeakerStimMethod) = "speakers_"*string(x.encoding)
sources(::SpeakerStimMethod) =
    ["male", "fem1", "fem2", "male-fem1-fem2"],
        ["male", "fem1", "fem2", "male-fem1-fem2", "male_other"]
train_source_indices(::SpeakerStimMethod) = (1,2,3,4,1)
function load_source_fn(method::SpeakerStimMethod,stim_events,fs,stim_info;
    test=false)

    function(i,j)
        if j <= 3
            load_speaker(stim_events,fs,i,j,
                encoding=method.encoding)
        # elseif j == 4
        #     load_speaker_mix_minus(stim_events,fs,i,1,
        #         encoding=method.encoding)
        elseif j == 4
            load_separated_speakers(stim_events,fs,i,
                encoding=method.encoding)
        elseif j == 5
            if !test
                load_speaker(stim_events,fs,i,1,
                    encoding=method.encoding)
            else
                load_other_speaker(stim_events,fs,stim_info,i,1,
                    encoding=method.encoding)
            end
        else
            error("Did not expect j == $j.")
        end
    end
end

Base.@kwdef struct ChannelStimMethod <: StimMethod
    encoding::Symbol
end
label(x::ChannelStimMethod) = "channels_"*string(x.encoding)
sources(::ChannelStimMethod) =
    ["left", "right"], ["left", "right", "left_other"]
train_source_indices(::ChannelStimMethod) = (1,2,1)
function load_source_fn(method::ChannelStimMethod,stim_events,fs,stim_info)
    function(i,j)
        if j <= 2
            load_channel(stim_events,fs,i,j,
                encoding=method.encoding)
        else
            load_other_channel(stim_events,fs,stim_info,i,1,
                encoding=method.encoding)
        end
    end
end


function encode_cache(body,key,stim_num)
    if key ∈ keys(encodings)
        encodings[key], stim_num
    else
        result = body()
        encodings[key] = result
        result, stim_num
    end
end

function load_speaker_mix_minus(events,tofs,stim_i,nosource_i;
        encoding=RMSEncoding())

    stim_num = events.sound_index[stim_i]
    encode_cache((:mix_minus,tofs,stim_num,nosource_i,encoding),stim_num) do
        fs = 0.0
        sources = map(setdiff(1:3,nosource_i)) do source_i
            x,fs = load(joinpath(stimulus_dir(),"mixtures","testing","mixture_components",
            @sprintf("trial_%02d_%1d.wav",stim_num,source_i)))

            x
        end
        minlen = mapreduce(x -> size(x,1),min,sources)
        mix = zeros(minlen)
        for t in eachindex(mix)
            for s in 1:length(sources)
                for c in 1:size(sources[s],2)
                    mix[t] += sources[s][t,c]
                end
            end
        end

        target_time = events.target_source[stim_i] ∈ [2,3] ?
            events.target_time[stim_i] : nothing
        encode(Stimulus(mix,fs,nothing,target_time),tofs,encoding)
    end
end

function load_speaker_mix(events,tofs,stim_i;encoding=RMSEncoding())
    stim_num = events.sound_index[stim_i]
    encode_cache((:mix,tofs,stim_num,encoding),stim_num) do
        x,fs = load(joinpath(stimulus_dir(),"mixtures","testing",
            @sprintf("trial_%02d.wav",stim_num)))
        if size(x,2) > 1
            x = sum(x,dims=2)
        end
        target_time = events.target_time[stim_i]
        encode(Stimulus(x,fs,nothing,target_time),tofs,encoding)
    end
end

function load_separated_speakers(events,tofs,stim_i;encoding=RMSEncoding())
    stim_num = events.sound_index[stim_i]
    encode_cache((:separated,tofs,stim_num,encoding),stim_num) do
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
            target_time = events.target_time[stim_i]
            encode(Stimulus(x,fs,file,target_time),tofs,encoding)
        end
    end
end

function load_speaker(events,tofs,stim_i,source_i;encoding=RMSEncoding())
    stim_num = events.sound_index[stim_i]
    target_time = events.target_source[stim_i] == source_i ?
        events.target_time[stim_i] : nothing
    load_speaker_(tofs,stim_num,source_i,target_time,encoding)
end

function load_speaker_(tofs,stim_num,source_i,target_time,encoding)
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

function load_other_speaker(events,tofs,info,stim_i,source_i;
    encoding=RMSEncoding())

    stim_num = events.sound_index[stim_i]
    stimuli = info["test_block_cfg"]["trial_sentences"]
    sentence_num = stimuli[stim_num][source_i]
    selected = rand(filter(i -> stimuli[i][source_i] != sentence_num,
        1:length(stimuli)))

    target_time = events.target_source[stim_i] == source_i ?
        events.target_time[stim_i] : nothing
    result, real_stim_num =
        load_speaker_(tofs,selected,source_i,target_time,encoding)
    result, stim_num
end

function load_channel(events,tofs,stim_i,source_i;encoding=:rms)
    stim_num = events.sound_index[stim_i]
    load_channel_(tofs,stim_num,source_i,encoding)
end

function load_channel_(tofs,stim_num,source_i,encoding)
    @assert source_i ∈ [1,2]
    encode_cache((:channel,tofs,stim_num,source_i,encoding),stim_num) do
        x,fs = load(joinpath(stimulus_dir(),"mixtures","testing",
            @sprintf("trial_%02d.wav",stim_num)))
        result = encode(Stimulus(x[:,source_i],fs,nothing),tofs,encoding)
    end
end

function load_other_channel(events,tofs,info,stim_i,source_i;
    encoding=:rms)

    stim_num = events.sound_index[stim_i]
    n = length(info["test_block_cfg"]["trial_sentences"])
    selected = rand(setdiff(1:n,stim_num))

    load_channel_(tofs,selected,source_i,encoding)
end
