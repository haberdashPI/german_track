
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



