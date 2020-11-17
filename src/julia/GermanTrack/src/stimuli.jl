using EEGCoding
using SignalOperators
using DataFrames
const encodings = Dict{Any, Array{Float64}}()
export SpeakerStimMethod, joint_source, male_source, fem1_source, fem2_source,
    other, mixed_sources, fem_mix_sources, JointSource, load_stimulus


function load_behavioral_stimulus_metadata()
    bdir = joinpath(raw_datadir(), "behavioral")
    info = joinpath(processed_datadir(), "behavioral", "stimuli", "stim_info.csv") |>
        CSV.read
    switch_times_file = joinpath(raw_datadir(), "behavioral", "switch_times.txt")
    salience_file = joinpath(processed_datadir(), "behavioral", "stimuli", "target_salience.csv")
    target_salience = .-CSV.read(salience_file).salience

    ## the below value is copied from an old config file (also the same value that's
    ## present in config.json, but if that ever changes, this shouldn't change)
    switch_length   = 1.2

    return (
        switch_length   = switch_length,
        trial_lengths   = fill(10, 50),
        target_times    = info.target_time,
        switch_regions    = @_(CSV.read(switch_times_file, header=0, delim=' ') |>
                                eachrow |> map(collect ∘ skipmissing,__) |>
                                map(tuple.(_1, _1 .+ 1), __) |>
                                map(vcat((0.0, switch_length/2), _), __)),
        directions      = info.direction,
        speakers        = info.speaker,
        target_salience = target_salience
    ) |> derived_metadata
end

critical_times_to_switch_regions(critical_times, switch_length) =
    @_ critical_times[2:(end-1)] |> zip(__[1:2:end], __[2:2:end]) |> collect |>
        # there is an initial 'degenerate' switch at the start of the trial
        # that is half the length of the other switches
        vcat((0.0, critical_times[1] / 2), __)

function load_stimulus_metadata(
    stim_filename = nothing,
    trial_lengths_filename = nothing,
    target_salience_filename = nothing)

    stim_filename = isnothing(stim_filename) ?
        joinpath(stimulus_dir(), "config.json") :
        stim_filename
    trial_lengths_filename = isnothing(trial_lengths_filename) ?
        joinpath(stimulus_dir(), "stimuli_lengths.csv") :
    target_salience_filename = isnothing(trial_lengths_filename) ?
        joinpath(stimulus_dir(), "target_salience.csv") :
        trial_lengths_filename

    # load and organize metadata about the stimuli
    open(stim_filename, read = true) do stream
        stim_info = JSON3.read(stream)
        critical_times    = map(times -> times ./ stim_info.fs,
            stim_info.test_block_cfg.switch_times)
        switch_length    = stim_info.switch_len

        return (
            switch_length    = switch_length,
            trial_lengths    = CSV.File(trial_lengths_filename).sound_length,
            speakers         = stim_info.test_block_cfg.trial_target_speakers,
            directions       = stim_info.test_block_cfg.trial_target_dir,
            target_times     = stim_info.test_block_cfg.target_times,
            critical_times   = critical_times,
            switch_regions   = critical_times_to_switch_regions.(critical_times,
                                    switch_length),
            target_salience  = CSV.File(joinpath(stimulus_dir(),
                                    "target_salience.csv")).salience
        ) |> derived_metadata
    end
end

function derived_metadata(meta)
    @assert meta.switch_regions |> length in [40,50]

    args = zip(meta.switch_regions, meta.target_times#= , meta.critical_times =#)
    switch_distance = map(args) do (switches, target#= , critical =#)
        if target == 0
            return missing
        end
        before = @_ switchdiff.(switches, target) |> filter(_ >= 0, __)
        if isempty(before)
            Inf
        else
            minimum(before)
        end
    end

    return (;meta...,

        switch_distance = switch_distance,

        salience_4level = begin
            quants = quantile(meta.target_salience, [0.25,0.5,0.75])
            sum(meta.target_salience .< quants', dims = 2) .+ 1
        end,

        # define some useful categories for these stimuli,
        salience_label = begin
            med = median(meta.target_salience)
            ifelse.(meta.target_salience .< med, "low", "high")
        end,

        target_time_label = begin
            early = @_ DataFrame(
                time = meta.target_times[1:length(meta.switch_regions)],
                switches = meta.switch_regions,
                row = 1:length(meta.switch_regions)) |>
            map(sum(switchdiff.(_1.switches, _1.time) .>= 0) <= 2 ? "early" : "late", eachrow(__))
        end,

        target_switch_label = begin
            med = @_ skipmissing(switch_distance) |> filter(!isinf,__) |> quantile(__, 0.5)
            map(switch_distance) do dist
                ismissing(dist) && return missing
                dist <= med ? "near" : "far"
            end
        end
    )
end

switchdiff(region, time) = time .- region[1]

abstract type StimMethod
end

abstract type AbstractSource
end
fortraining(x::AbstractSource) = x

const RowType = Union{DataFrameRow, NamedTuple}

function encode_cache(body, key, stim_num)
    if key ∈ keys(encodings)
        encodings[key], stim_num
    else
        result = body()
        encodings[key] = result
        result, stim_num
    end
end
clear_stim_cache!() = empty!(encodings)

function load_stimulus(source::AbstractSource, stim_i::Int, encoding, events, tofs,
    stim_info)

    load_stimulus(source, events[stim_i, :], encoding, tofs, stim_info)
end

struct SingleSource <: AbstractSource
    name::String
    index::Int
end
Base.string(x::SingleSource) = x.name
male_source = SingleSource("male", 1)
fem1_source = SingleSource("fem1", 2)
fem2_source = SingleSource("fem2", 3)

function load_stimulus(source::SingleSource, event::RowType, encoding, tofs, stim_info)
    stim_num = event.sound_index
    target_time = event.target_source == source.index ?
        event.target_time : nothing
    load_single_speaker(tofs, stim_num, source.index, target_time,
        encoding)
end

function load_single_speaker(tofs, stim_num, source_i, target_time, encoding)
    encode_cache((:speaker, tofs, stim_num, source_i, encoding), stim_num) do
        file = joinpath(stimulus_dir(), "mixtures", "testing", "mixture_components",
            @sprintf("trial_%02d_%1d.wav", stim_num, source_i))
        x, fs = load(file)
        if size(x, 2) > 1
            x = sum(x, dims = 2)
        end
        encode(Stimulus(x, fs, file, target_time), tofs, encoding)
    end
end

struct JointSource <: AbstractSource
    collapse_dims::Bool
end
joint_source = JointSource(true)
Base.string(::JointSource) = "joint"

function load_stimulus(source::JointSource, event::RowType, encoding,
    tofs, stim_info)

    stim_num = event.sound_index
    target_time = event.target_time
    load_joint_stimulus(tofs, stim_num, target_time,
        encoding, source.collapse_dims)
end

adddim(x) = reshape(x, 1, size(x)...)
function load_joint_stimulus(tofs, stim_num, target_time, encoding, collapse)

    encode_cache((:joint, tofs, stim_num, encoding, collapse), stim_num) do
        fs = 0
        stimdir = joinpath(stimulus_dir(), "mixtures", "testing",
            "mixture_components")
        sources = (joinpath(stimdir, @sprintf("trial_%02d_%1d.wav", stim_num, j))
            for j in 1:3)
        function encodefile(file)
            x, fs = load(file)
            if size(x, 2) > 1
                x = sum(x, dims = 2)
            end
            encode(Stimulus(x, fs, file, target_time), tofs, encoding)
        end
        if collapse
            mapreduce(encodefile, hcat, sources)
        else
            mapreduce(adddim ∘ encodefile, vcat, sources)
        end
    end
end

struct OtherSource{S} <: AbstractSource
    source::S
end
other(x::AbstractSource) = OtherSource(x)
other(x::OtherSource) = error("Already 'othered'")
fortraining(x::OtherSource) = x.source
Base.string(x::OtherSource) = string("other_", string(x.source))

function load_stimulus(source::OtherSource{JointSource}, event::RowType,
    encoding, tofs, info)

    stim_num = event.sound_index
    selected = rand(setdiff(1:50, stim_num))

    target_time = event.target_time
    result, real_stim_num =
        load_joint_stimulus(tofs, selected, target_time,
            encoding, source.source.collapse_dims)
    result, stim_num
end

function load_stimulus(other::OtherSource{SingleSource}, event::RowType,
    encoding, tofs, info)

    stim_num = event.sound_index
    stimuli = info["test_block_cfg"]["trial_sentences"]
    sentence_num = stimuli[stim_num][other.source.index]
    selected = rand(filter(i -> stimuli[i][other.source.index] != sentence_num,
        1:length(stimuli)))

    target_time = event.target_source == other.source.index ?
        event.target_time : nothing
    result, real_stim_num =
        load_single_speaker(tofs, selected, other.source.index, target_time,
            encoding)
    result, stim_num
end

struct MixedSources <: AbstractSource
    indices::Vector{Int}
    name::String
end
Base.string(x::MixedSources) = x.name
mixed_sources = MixedSources(1:3, "all")
fem_mix_sources = MixedSources(2:3, "fem1+fem2")

function load_stimulus(mixed::MixedSources, event::RowType, encoding,
    tofs, info)

    stim_num = event.sound_index
    key = (:mixed, mixed.indices, stim_num, tofs, stim_num, encoding)
    encode_cache(key, stim_num) do
        filenames = map(mixed.indices) do source_i
            joinpath(stimulus_dir(), "mixtures", "testing",
                "mixture_components",
                @sprintf("trial_%02d_%1d.wav", stim_num, source_i))
        end
        mixture, fr = Mix(filenames...) |> ToChannels(1) |> sink

        target_time = event.target_source ∈ mixed.indices ?
            event.target_time : nothing
        encode(Stimulus(mixture, fr, nothing, target_time),
            tofs, encoding)
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
