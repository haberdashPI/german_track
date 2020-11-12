export read_eeg_binary, read_mcca_proj, load_subject, events_for_eeg, sidfor,
    load_directions, load_all_subjects, processed_datadir, raw_datadir,
    stimulus_dir, raw_stim_dir

# we save cached results in JSON fromat
DrWatson._wsave(file, data::Dict) = open(io -> JSON3.write(io, data), file, "w")

function mkpathfile(args...)
    if length(args) == 0
        return ""
    end

    if occursin(r"\.[a-z]+$", args[end])
        if length(args) > 1
            mkpath(joinpath(args[1:(end-1)]...))
        end
    else
        mkpath(joinpath(args...))
    end

    joinpath(args...)
end

"""
    processed_datadir(subdir1,subdir2,....)

Get (and possibly create) a directory for processed data.
"""
processed_datadir(args...) =
    mkpathfile(joinpath(datadir(), "processed", args...))
"""
    raw_datadir(subdir1,subdri2,...)

Get a directory of raw data.
"""
raw_datadir(args...) = joinpath(datadir(), "raw", args...)

"""
    stimulus_dir()

Get the directory where processed stimuli data are stored.
"""
stimulus_dir(args...) = processed_datadir("stimuli", args...)

"""
    raw_stim_dir()

Get the directory where raw stimuli data are stored.
"""
raw_stim_dir() = raw_datadir("stimuli")


"""
    read_eeg_binary(filename)

Helper function that loads binary EEG data stored in custom format.

See also [`load_subject`](#).
"""
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

"""
    read_mcca_proj(filename)

Helper function, load mcca projects from custom binary format.

See also [`load_subject`](#).
"""
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

"""
    read_h5_subj(filename)

Helper function. Load HDF5 formated subject data.

See also [`load_subject`](#).
"""
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

# Store subject data in a cache for easy loading later on.
const subject_cache = Dict()
Base.@kwdef struct SubjectData
    eeg::EEGData
    events::DataFrame
end

"""
    load_all_subjects(dir, ext, metadata = load_stimulus_metadata())

Load all subjects located under `dir` with extension `ext`. This includes
a comperhensive dictionary from subject ids to all subject data and an aggregate
dataframe of all event data.

Caches subject loading to speed it up.
"""
function load_all_subjects(dir, ext, stim_info = load_stimulus_metadata())
    eeg_files = dfhit = @_ readdir(dir) |> filter(endswith(_, string(".",ext)), __)
    subjects = Dict(
        sidfor(file) => load_subject(
            joinpath(dir, file), stim_info,
            encoding = RawEncoding()
        ) for file in eeg_files)
    events = @_ mapreduce(_.events, append!!, values(subjects))

    subjects, events
end

"""
    load_subject(file, metadata = load_stimulus_metadata(); encoding = RawEncoding(),
        framerate = missing)

Load the given subject, encoding the EEG data acording to `encoding` (which by default
just uses the raw data). The variable `metdata` must contain the stimulus meta-data.
It can be loaded using `load_stimulus_metadata`.
"""
function load_subject(file, stim_info = load_stimulus_metadata();
    encoding = RawEncoding(), framerate = missing)

    if !isfile(file)
        error("File '$file' does not exist.")
    end

    stim_events = events_for_eeg(file, stim_info)

    data = get!(subject_cache, (file, encoding, framerate)) do
        data = if endswith(file, ".mcca_proj") || endswith(file, ".mcca")
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

"""
    events(filename, metadata)

Load the given file of stimulus events; requires metadata loaded via
`load_stimulus_metadata`.
"""

function events(event_file, stim_info)
    stim_events = DataFrame(CSV.File(event_file))

    source_names =

    # columns that are determined by the stimulus (and thus derived using the index of the
    # stimulus: sound_index)
    si = stim_events.sound_index
    info(sym) = get.(Ref(getproperty(stim_info, sym)), si, missing)

    target_times = stim_info.target_times
    stim_events[!, :target_present]      .= target_times[si] .> 0
    stim_events[!, :target_source]        = get.(Ref(["male", "fem1", "fem2"]),
                                                 Int.(stim_info.speakers[si]), missing)
    stim_events[!, :target_time]          = ifelse.(target_times[si] .> 0,
                                                    target_times[si], missing)
    stim_events[!, :sid]                 .= sidfor(event_file)
    stim_events[!, :trial_index]         .= 1:size(stim_events, 1)
    stim_events[!, :salience]            .= info(:target_salience)
    stim_events[!, :direction]           .= info(:directions)
    stim_events[!, :target_switch_label] .= info(:target_switch_label)
    stim_events[!, :salience_label]      .= info(:salience_label)
    stim_events[!, :salience_4level]     .= info(:salience_4level)
    stim_events[!, :target_time_label]   .= info(:target_time_label)
    stim_events[!, :switch_regions]      .= info(:switch_regions)
    stim_events[!, :trial_length]        .= info(:trial_lengths)
    stim_events[!, :switch_distance]     .= info(:switch_distance)

    stim_events
end

"""
    events_for_eeg(filename, metadata)

Like `events` but first translate the filename, which should be an eeg file,
to the corresponding CSV file of events.
"""
function events_for_eeg(file, stim_info)
    sid = sidfor(file)
    @_ joinpath(processed_datadir("eeg"), @sprintf("sound_events_%03d.csv", sid)) |>
        events(__, stim_info)
end

"""
    sidfor(filename)

Using the naming conventions of this project, extact the subject ID from the given filename.
"""
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

"""
    load_directions(file)

Read a file with phase directions for three sources, with the extenion `.dir` (by
convention). These are metadata that indicate the location of each speaker in space as a
phase value, sampled at a given framerate.
"""
function load_directions(file)
    open(file, read = true) do stream
        framerate = read(stream, Float64)
        len1 = read(stream, Int)
        len2 = read(stream, Int)
        len3 = read(stream, Int)

        dir1 = reinterpret(Float64, read(stream, sizeof(Float64)*len1))
        dir2 = reinterpret(Float64, read(stream, sizeof(Float64)*len2))
        dir3 = reinterpret(Float64, read(stream, sizeof(Float64)*len3))

        @assert length(dir1) == len1
        @assert length(dir2) == len2
        @assert length(dir3) == len3

        Directions(dir1, dir2, dir3, framerate)
    end
end
