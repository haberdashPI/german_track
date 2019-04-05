
function load_subject(file)
    mf = MatFile(file)
    data = get_mvariable(mf,:dat)
    close(mf)

    sid = parse(Int,match(r"([0-9]+)(_ica)?\.mat$",file)[1])

    fdir, _ = splitdir(file)
    event_file = joinpath(fdir,@sprintf("sound_events_%03d.csv",sid))
    stim_events = DataFrame(load(event_file))

    data,stim_events,sid
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

function cachefn(prefix,fn,args...)
    file = joinpath(cache_dir,prefix * ".jld2")
    if isfile(file)
        load(file,"contents")
    else
        result = fn(args...)
        save(file,"contents",result)
        result
    end
end
