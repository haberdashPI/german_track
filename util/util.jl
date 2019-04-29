
function load_subject(file,stim_info)
    mf = MatFile(file)
    data = get_mvariable(mf,:dat)
    close(mf)
    stim_events, sid = events_for_eeg(file,stim_info)

    data,stim_events,sid
end

function events_for_eeg(file,stim_info)
    matched = match(r"eeg_response_([0-9]+)(_[a-z_]+)?([0-9]+)?\.mat$",file)
    sid = parse(Int,matched[1])

    event_file = joinpath(data_dir,@sprintf("sound_events_%03d.csv",sid))
    stim_events = DataFrame(load(event_file))

    target_times = convert(Array{Float64},
        stim_info["test_block_cfg"]["target_times"][stim_events.sound_index])

    # derrived columns
    stim_events[:target_time] = target_times
    stim_events[:target_present] = target_times .> 0
    stim_events[:correct] = stim_events.target_present .==
        (stim_events.response .== 2)
    stim_events[:,:bad_trial] = convert.(Bool,stim_events.bad_trial)

    stim_events, sid
end


function load_sentence(events,info,stim_i,source_i)
    stim_num = events.sound_index[stim_i]
    x,fs = load(joinpath(stimulus_dir,"mixtures","testing","mixture_components",
        @sprintf("trial_%02d_%1d.wav",stim_num,source_i)))
    SampleBuf(x,fs)
end

function load_other_sentence(events,info,stim_i,source_i)
    stim_num = events.sound_index[stim_i]
    sentences = info["test_block_cfg"]["trial_sentences"][:,source_i]
    # randomly select one of the stimuli != stim_i
    selected = rand(vcat(1:stim_num-1,stim_num+1:length(sentences)))

    x,fs = load(joinpath(stimulus_dir,"mixtures","testing","mixture_components",
        @sprintf("trial_%02d_%1d.wav",selected,source_i)))
    SampleBuf(x,fs)
end

function cachefn(prefix,fn,args...;oncache=() -> nothing,kwds...)
    file = joinpath(cache_dir,prefix * ".jld2")
    if isfile(file)
        oncache()
        load(file,"contents")
    else
        result = fn(args...;kwds...)
        save(file,"contents",result)
        result
    end
end

function folds(k,indices)
    len = length(indices)
    fold_size = ceil(Int,len / k)
    map(1:k) do fold
        test = indices[((fold-1)fold_size+1) : (min(len,fold*fold_size))]
        train = setdiff(indices,test)

        (train,test)
    end
end

# TODO: select the switch areas
function only_switches(switches,max_time;window=(-0.250,0.250))
    result = Array{Tuple{Float64,Float64}}(undef,length(switches))

    i = 0
    stop = 0
    for switch in switches
        new_stop = min(switch+window[2],max_time)
        if stop < switch+window[1]
            i = i+1
            result[i] = (switch+window[1],new_stop)
        elseif i > 0
            result[i] = (result[i][1], new_stop)
        else
            i = i+1
            result[i] = (0,new_stop)
        end
        stop = new_stop
    end

    view(result,1:i)
end


function remove_switches(switches,max_time;wait_time=0.5)
    result = Array{Tuple{Float64,Float64}}(undef,length(switches)+1)

    start = 0
    i = 0
    for switch in switches
        if start < switch
            i = i+1
            result[i] = (start,switch)
        end
        start = switch+wait_time
    end
    if start < max_time
        i = i+1
        result[i] = (start,max_time)
    end

    view(result,1:i)
end
