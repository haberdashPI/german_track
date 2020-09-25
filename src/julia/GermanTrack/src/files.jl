export read_eeg_binary, read_mcca_proj, load_subject, events_for_eeg, sidfor,
    load_directions, load_all_subjects

function read_eeg_binary(filename)
    open(filename) do file
        # number of channels
        nchan = read(file, Int32)
        # channels names
        channels = Vector{String}(undef, nchan)
        for i in 1:nchan
            len = read(file, Int32)
            channels[i] = String(read(file, len))
        end
        # number of trials
        ntrials = read(file, Int32)
        # sample rate
        fs = read(file, Int32)
        # trials
        trials = Vector{Array{Float64}}(undef, ntrials)
        for i in 1:ntrials
            # trial size
            row = read(file, Int32)
            col = read(file, Int32)
            # trial
            trial = Array{Float64}(undef, row, col)
            read!(file, trial)
            trials[i] = trial
        end

       EEGData(data = trials, label = channels, fs = fs)
    end
end

function read_mcca_proj(filename)
    @info "Reading projected components"
    open(filename) do file
        # number of channels
        nchan = read(file, Int32)
        if nchan > 4096
            error("Unexpected number of channels: $(nchan)")
        end
        # channels names
        channels = Vector{String}(undef, nchan)
        for i in 1:nchan
            len = read(file, Int32)
            channels[i] = String(read(file, len))
        end
        # number of components
        ncomp = read(file, Int32)
        # components
        comp = Array{Float64}(undef, ncomp, nchan)
        read!(file, comp)
        # number of trials
        ntrials = read(file, Int32)
        # sample rate
        fs = read(file, Int32)
        # projected trials
        trials = Vector{Array{Float64}}(undef, ntrials)
        for i in 1:ntrials
            # trial size
            row = read(file, Int32)
            col = read(file, Int32)
            # trial
            trial = Array{Float64}(undef, row, col)
            read!(file, trial)
            trials[i] = trial #(trial'comp)'
        end

       EEGData(data = trials, label = channels, fs = fs)
    end
end

function read_h5_subj(filename)
    h5open(filename, "r") do file
        channels   = read(file, "channels")
        components = read(file, "components")
        ntrials    = only(read(file, "trials/count"))
        samplerate = only(read(file, "trials/samplerate"))

        trials = Vector{Array{Float64, 2}}(undef, ntrials)
        for i in 1:ntrials
            trials[i] = read(file, @sprintf("trials/%03d", i))
        end

        EEGData(data = trials, label = channels, fs = samplerate)
    end
end

const subject_cache = Dict()
Base.@kwdef struct SubjectData
    eeg::EEGData
    events::DataFrame
end

function load_all_subjects(dir, ext)
    eeg_files = dfhit = @_ readdir(dir) |> filter(endswith(_, string(".",ext)), __)
    subjects = Dict(
        sidfor(file) => load_subject(
            joinpath(dir, file), stim_info,
            encoding = RawEncoding()
        ) for file in eeg_files)
    events = @_ mapreduce(_.events, append!!, values(subjects))

    subjects, events
end

function load_subject(file, stim_info;encoding = RawEncoding(), framerate = missing)
    if !isfile(file)
        error("File '$file' does not exist.")
    end

    stim_events = events_for_eeg(file, stim_info)

    data = get!(subject_cache, (file, encoding, framerate)) do
        # data = if endswith(file, ".mat")
        #     mf = MatFile(file)
        #     get_mvariable(mf, :dat)
        data = if endswith(file, ".bson")
            @load file data
            data
        elseif endswith(file, ".mcca_proj") || endswith(file, ".mcca")
            read_mcca_proj(file)
        elseif endswith(file, ".eeg")
            read_eeg_binary(file)
        elseif endswith(file, ".h5")
            read_h5_subj(file)
        else
            pat = match(r"\.([^\.]+)$", file)
            if !isnothing(pat)
                ext = pat[1]
                error("Unsupported data format '.$ext'.")
            else
                error("Unknown file format for $file")
            end
        end

        encode(data, framerate, encoding)
    end

    SubjectData(eeg = data, events = stim_events)
end

function events(event_file, stim_info)
    stim_events = DataFrame(CSV.File(event_file))

    target_times = convert(Array{Float64}, stim_info["test_block_cfg"]["target_times"])
    source_indices = convert(Array{Float64},
        stim_info["test_block_cfg"]["trial_target_speakers"])
    source_names = ["male", "fem1", "fem2"]

    # columns that are determined by the stimulus (and thus derived using the index of the
    # stimulus: sound_index)
    si = stim_events.sound_index
    stim_events[!, :target_source] = get.(Ref(source_names), Int.(source_indices[si]),
        missing)
    stim_events[!, :target_present] .= target_times[si] .> 0
    stim_events[!, :target_time] = ifelse.(target_times[si] .> 0, target_times[si], missing)
    if :bad_trial ∈ propertynames(stim_events)
        stim_events[!, :bad_trial] = convert.(Bool, stim_events.bad_trial)
    else
        @warn "Could not find `bad_trial` column in file '$event_file'."
        stim_events[!, :bad_trial] .= false
    end
    stim_events.sid = sid
    stim_events.trial_index = 1:size(stim_events, 1)
    stim_events.salience = get.(Ref(target_salience), si, missing)
    stim_events.direction = get.(Ref(directions), si, missing)
    stim_events.target_switch_label = get.(Ref(target_switch_label), si, missing)
    stim_events.salience_label = get.(Ref(salience_label), si, missing)
    stim_events.salience_4level = get.(Ref(salience_4level), si, missing)
    stim_events.target_time_label = get.(Ref(target_time_label), si, missing)

    stim_events
end

function events_for_eeg(file, stim_info)
    sid = sidfor(file)
    @_ joinpath(processed_datadir("eeg"), @sprintf("sound_events_%03d.csv", sid)) |>
        events(__, stim_info)
end

function sidfor(filepath)
    file = splitdir(filepath)[2]
    pattern = r"eeg_response_([0-9]+)(_[a-z_]+)?([0-9]+)?(_unclean)?\.[a-z_0-9]+$"
    matched = match(pattern, file)
    if isnothing(matched)
        pattern = r"([0-9]+).*\.[a-z_0-9]+$"
        matched = match(pattern, file)
        if isnothing(matched)
            error("Could not find subject id in filename '$file'.")
        end
    end
    parse(Int, matched[1])
end


struct Directions
    dir1::Vector{Float64}
    dir2::Vector{Float64}
    dir3::Vector{Float64}
    framerate::Float64
end

function load_directions(file)
    open(file, read = true) do stream
        framerate = read(stream, Float64)
        len1 = read(stream, Int)
        len2 = read(stream, Int)
        len3 = read(stream, Int)

        dir1 = reinterpret(Float64, read(stream, sizeof(Float64)*len1))
        dir2 = reinterpret(Float64, read(stream, sizeof(Float64)*len1))
        dir3 = reinterpret(Float64, read(stream, sizeof(Float64)*len1))

        @assert length(dir1) == len1
        @assert length(dir2) == len2
        @assert length(dir3) == len3

        Directions(dir1, dir2, dir3, framerate)
    end
end


