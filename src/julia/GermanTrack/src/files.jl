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
function load_all_subjects(dir, ext, stim_info = load_stimulus_metadata();
    encoding = RawEncoding(), framerate = missing)

    eeg_files = dfhit = @_ readdir(dir) |> filter(endswith(_, string(".",ext)), __)
    subjects = Dict(
        sidfor(file) => load_subject(
            joinpath(dir, file), stim_info,
            encoding = encoding,
            framerate = framerate
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


function _parse_cache_args(prefix, args)
    # 2. symbols to cache
    cache_symbols(x) = error("Unexpected expression `$(x)`")

    # 2a. a bare variable name
    cache_symbols(x::Symbol) = x
    cache_filetypes(x::Symbol) = :arrow

    # 2b. tuple of (varname, :filetype)
    function cache_symbols(ex::Expr)
        if isexpr(ex, :tuple)
            @capture(ex, (var_, type_))
            return var
        else
            error("Unexpected expression `$(ex)`.")
        end
    end
    function cache_filetypes(ex::Expr)
        @capture(ex, (var_, type_))
        if type isa QuoteNode
            return type.value
        else
            error("Unexpected expression `$(type)`.")
        end
    end
    symbols = cache_symbols.(args[1:end])
    file_types = cache_filetypes.(args[1:end])

    checktypes(x::Symbol) = x ∈ [:arrow, :bson, :jld] ||
        error("Unexpected filetype `$(x)``.")
    checktypes.(file_types)

    symbols, file_types
end

_fnames(prefix, symbols, types) =
    :(string.($(esc(prefix)), "-", $(string.(symbols)), ".", $(string.(types))))
_fname(prefix, symbols, types, i) =
    :(string($(esc(prefix)), "-", $(string(symbols[i])), ".", $(string(types[i]))))

"""
    GermanTrack.@use_cache prefix variables... begin
        body
    end

Store the listed variables (defined in `body`) to files. They can be stored as Arrow
(default), JLD or BSON files. To speciy the file type give the variable in a tuple
with `:arrow`, `:bson` or `:jld` (e.g. `(my_flux_model, :bson)`).

This is used to cache long-running code, to avoid unncessary re-calculations;
if the needed files already exists, the body of the macro will not be re-run.
"""
macro use_cache(prefix, args...)
    body = args[end]
    symbols, types = _parse_cache_args(prefix, args[1:(end-1)])

    #### Verify that each variable listed in cache header exists in the body
    # of the macro

    # check for all symbols before running the code
    # (avoids getting the error after running some long-running piece of code)
    found_symbols = Set{Symbol}()
    MacroTools.postwalk(body) do expr
        if expr isa Symbol && expr ∈ symbols
            push!(found_symbols, expr)
        end
        expr
    end
    missing_index = @_ findfirst(_ ∉ found_symbols, symbols)

    if !isnothing(missing_index)
        error("Could not find symbol `$(symbols[missing_index])` in cache body, "*
            "check spelling.")
    end

    #### code generation
    quote
        begin
            # run body
            if all(isfile, $(_fnames(prefix, symbols, types)))
                $(_load_cache(prefix, symbols, types))
            else # create the values
                $(esc(body))
                $(_save_cache(prefix, symbols, types))
            end
            nothing
        end
    end
end

"""
    GermanTrack.@save_cache prefix variables...

Store data as per [`@use_cache`](@ref), but the variables should already be defined.
"""
macro save_cache(prefix, args...)
    symbols, types = _parse_cache_args(prefix, args)
    _save_cache(prefix, symbols, types)
end

"""
    GermanTrack.@load_cache prefix variables...

Deine variables, loading them from files, as per [`@use_cache`](@ref).
"""
macro load_cache(prefix, args...)
    symbols, types = _parse_cache_args(prefix, args)
    _load_cache(prefix, symbols, types)
end

function _load_cache(prefix, symbols, types)
    quote
        $(map(enumerate(zip(symbols, types))) do (i, (var, type))
            if type == :arrow
                quote
                    $(esc(var)) =
                        DataFrame(Arrow.Table($(_fname(prefix, symbols, types, i))))
                    @info string("Loaded ", $(string(var)), " from ",
                        $(_fname(prefix, symbols, types, i)))
                end
            elseif type == :bson
                quote
                    $(esc(var)) = load($(_fname(prefix, symbols, types, i)))["data"]
                    @info string("Loaded ", $(string(var)), " from ",
                        $(_fname(prefix, symbols, types, i)))
                end
            elseif type == :jld
                quote
                    $(esc(var)) = load($(_fname(prefix, symbols, types, i)), "data")
                    @info string("Loaded ", $(string(var)), " from ",
                        $(_fname(prefix, symbols, types, i)))
                end
            else
                errror("Unexpected error: report a bug.")
            end
        end...)
    end
end

function _save_cache(prefix, symbols, types)
    quote
        # store code state
        state = tag!(Dict())

        @info "Saving data..."
        # store variables in files
        $(map(enumerate(zip(symbols, types))) do (i, (var, type))
            if type == :arrow
                quote
                    Arrow.setmetadata!($(esc(var)), convert(Dict{String, String}, state))
                    Arrow.write($(_fname(prefix, symbols, types, i)),
                        $(esc(var)), compress = :lz4)
                    @info string("Saved ", $(string(var))," to ",
                        $(_fname(prefix, symbols, types, i)))
                end
            elseif type == :bson
                quote
                    data = deepcopy(state)
                    data["data"] = $(esc(var))
                    save($(_fname(prefix, symbols, types, i)), data)
                    @info string("Saved ", $(string(var))," to ",
                        $(_fname(prefix, symbols, types, i)))
                end
            elseif type == :jld
                quote
                    save($(_fname(prefix, symbols, types, i)), "data",
                        $(esc(var)), "state", state)
                    @info string("Saved ", $(string(var))," to ",
                        $(_fname(prefix, symbols, types, i)))
                end
            else
                errror("Unexpected error: report a bug.")
            end
        end...)
    end
end
