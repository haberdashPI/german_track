
function load_subject(file)
    mf = MatFile(file)
    data = get_variable(mf,:dat)
    close(mf)

    sid = parse(Int,match(r"([0-9]+)(_ica)?\.mat$",file)[1])

    fdir, _ = splitdir(file)
    event_file = joinpath(fdir,@sprintf("sound_events_%03d.csv",sid))
    stim_events = DataFrame(load(event_file))

    data,stim_events,sid
end

function load_sentence(events,info,stim_i,source_i)
    stim_num = events.sound_index[stim_i]
    load(joinpath(stimulus_dir,"mixtures","testing","mixture_components",
        @sprintf("trial_%02d_%1d.wav",stim_num,source_i)))
end

function cachefn(file,fn,args...)
    if isfile(file)
        load(file,"contents")
    else
        result = fn(args...)
        save(file,"contents",result)
    end
end
