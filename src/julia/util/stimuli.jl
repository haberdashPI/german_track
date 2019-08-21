using EEGCoding
const encodings = Dict{Any,Array{Float64}}()

function load_speaker_mix_minus(events,tofs,stim_i,nosource_i;encoding=:rms)
    stim_num = events.sound_index[stim_i]
    key = (:mix_minus,tofs,stim_num,nosource_i,encoding)
    if key ∈ keys(encodings)
        encodings[key]
    else
        fs = 0.0
        sources = map(setdiff(1:3,nosource_i)) do source_i
            x,fs = load(joinpath(stimulus_dir,"mixtures","testing","mixture_components",
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
            events.target_time[stim_i] : missing
        result = encode_stimulus(SampleBuf(mix,fs),tofs,target_time,
            method=encoding)
        encodings[key] = result
        result
    end
end

function load_speaker_mix(events,tofs,stim_i;encoding=:rms)
    stim_num = events.sound_index[stim_i]
    key = (:mix,tofs,stim_num,encoding)
    if key ∈ keys(encodings)
        encodings[key]
    else
        x,fs = load(joinpath(stimulus_dir,"mixtures","testing",
            @sprintf("trial_%02d.wav",stim_num)))
        if size(x,2) > 1
            x = sum(x,dims=2)
        end
        target_time = events.target_time[stim_i]
        result = encode_stimulus(SampleBuf(x,fs),tofs,target_time,
            method=encoding)
        encodings[key] = result
        result
    end
end

function load_speaker(events,tofs,stim_i,source_i;encoding=:rms)
    stim_num = events.sound_index[stim_i]
    target_time = events.target_source[stim_i] == source_i ?
        events.target_time[stim_i] : missing
    load_speaker_(tofs,stim_num,source_i,target_time,encoding)
end

function load_speaker_(tofs,stim_num,source_i,target_time,encoding)
    key = (:speaker,tofs,stim_num,source_i,encoding)
    if key ∈ keys(encodings)
        encodings[key]
    else
        x,fs = load(joinpath(stimulus_dir,"mixtures","testing","mixture_components",
            @sprintf("trial_%02d_%1d.wav",stim_num,source_i)))
        if size(x,2) > 1
            x = sum(x,dims=2)
        end
        result = encode_stimulus(SampleBuf(x,fs),tofs,target_time,method=encoding)
        encodings[key] = result
        result
    end
end

function load_other_speaker(events,tofs,info,stim_i,source_i;
    encoding=:rms)

    stim_num = events.sound_index[stim_i]
    stimuli = info["test_block_cfg"]["trial_sentences"]
    sentence_num = stimuli[stim_num][source_i]
    selected = rand(filter(i -> stimuli[i][source_i] != sentence_num,
        1:length(stimuli)))

    target_time = events.target_source[stim_i] == source_i ?
        events.target_time[stim_i] : missing
    load_speaker_(tofs,selected,source_i,target_time,encoding)
end

function load_channel(events,tofs,stim_i,source_i;encoding=:rms)
    stim_num = events.sound_index[stim_i]
    load_channel_(tofs,stim_num,source_i,encoding)
end

function load_channel_(tofs,stim_num,source_i,encoding)
    @assert source_i ∈ [1,2]
    key = (:channel,tofs,stim_num,source_i,encoding)
    if key ∈ keys(encodings)
        encodings[key]
    else
        x,fs = load(joinpath(stimulus_dir,"mixtures","testing",
            @sprintf("trial_%02d.wav",stim_num)))
        result = encode_stimulus(SampleBuf(x[:,source_i],fs),tofs,
            method=encoding)
        encodings[key] = result
        result
    end
end

function load_other_channel(events,tofs,info,stim_i,source_i;
    encoding=:rms)

    stim_num = events.sound_index[stim_i]
    n = length(info["test_block_cfg"]["trial_sentences"])
    selected = rand(setdiff(1:n,stim_num))

    load_channel_(tofs,selected,source_i,encoding)
end
