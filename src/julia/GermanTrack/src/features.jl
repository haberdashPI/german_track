export windowtarget, windowbaseline, windowswitch, compute_freqbins, window_target_switch

# Windowing Functions
# =================================================================

# Utility
# -----------------------------------------------------------------

const NUM_STIMULI = 50
const COND_ID = Dict(
    "global"  => 0,
    "object"  => 1,
    "spatial" => 2,
)
"""
    trialrng(id, event)

Random number genration as a function of the individual trials. The same random
number gets picked for the same stimulus and the same condition.
"""
function trialrng(id, event)
    seed = stablehash(id)
    Threefry2x((seed, NUM_STIMULI*COND_ID[event.condition] + event.sound_index))
end

"""
    windowtrial(trial, fs, (from, to))

Given a a window (from, to) in seconds, return the frames of `trial` in this range.
"""
function windowtrial(trial, fs, (from, to))
    start, stop = clamp.(round.(Int, fs.*(from, to)), 1, size(trial, 2))
    view(trial, :, start:stop)
end

# Window Selection
# -----------------------------------------------------------------

"""
    windowsat(times, max_time; from, to)

Find all possible windows starting at `from` and ending at `to`, relative to each of the
times in `times`, mering any overlapping windows.

In the simplest case, where no overlap occurs between adjacent windows, this is equivalent
to `map(x -> (from + x, to + x), times)`. When windows overlap, they are merged into a
single window.

"""
windowsat(time::Number, max_time; kwds...) = windowsat((time,), max_time; kwds...)[1]
function windowsat(times, max_time; from, to)
    result = Array{Tuple{Float64, Float64}}(undef, length(times))

    i = 0
    stop = 0
    for time in times
        new_stop = min(time + to, max_time)
        if stop < time + from
            i = i + 1
            result[i] = (time + from, new_stop)
        elseif i > 0
            result[i] = (result[i][1], new_stop)
        else
            i = i + 1
            result[i] = (0, new_stop)
        end
        stop = new_stop
    end

    view(result, 1:i)
end

"""
    windows_far_from(times, max_time; mindist, minlength)
"""
function windows_far_from(times, max_time; mindist, minlength)
    result = Array{Tuple{Float64, Float64}}(undef, length(times) + 1)
    start = 0
    i = 0
    for time in times
        if start < time
            if time - mindist - minlength > start
                i = i + 1
                result[i] = (start, time - mindist)
            end
            start = time + mindist
        end
    end
    if start < max_time
        i = i + 1
        result[i] = (start, max_time)
    end
    view(result, 1:i)
end

"""
    sample_from_ranges(rng, ranges)

Given a list of windows (array of 2-ary tuples), randomly select a time point within
one of these windows.
"""
function sample_from_ranges(rng, ranges)
    weights = Weights(map(x -> x[2] - x[1], ranges))
    range = StatsBase.sample(rng, ranges, weights)
    rand(Distributions.Uniform(range...))
end

# Windowing Functions
# -----------------------------------------------------------------

struct WindowFnBounds{S,T}
    start::Float64
    len::Float64
    entry::T
    name::String
end
function Base.show(io::IO, x::WindowFnBounds{S,<:Nothing}) where S
    print(io, S,
        " (start = ",@sprintf("%2.2f", x.start),", ",
        "len = ",@sprintf("%2.2f", x.len),")")
end

struct WindowFnFn{S,T}
    fn::Function
    entry::T
    name::String
end
function Base.show(io::IO, x::WindowFnFn{S,<:Nothing}) where S
    print(io, name, ": ", S, " (", string(fn), ")")
end

resolve(x::WindowFnBounds, _) = x
function resolve(x::WindowFnFn{S,T}, data) where {S,T}
    vals = x.fn(data)
    WindowFnBounds{S,T}(vals.start, vals.len, x.entry, x.name)
end

function WindowFn(Sym,entry = nothing; start=nothing, len=nothing,
    windowfn=nothing, name="window")

    if isnothing(start) && isnothing(len) && isnothing(windowfn)
        error("Need either `start` and `len` or `windowfn` keyword arguments.")
    elseif isnothing(windowfn)
        (isnothing(start) || isnothing(len)) && error("Need both `start` and `len` "*
            "keyword arguments.")
        WindowFnBounds{Sym, typeof(entry)}(start, len, entry, name)
    else
        !(isnothing(start) && isnothing(len)) && error("Cannot have both `windowfn` and ",
            "one of `start` and `len` keyword arguments.")
        WindowFnFn{Sym, typeof(entry)}(windowfn, entry, name)
    end
end

"""
    windowtarget(;start, len)
    windowtarget(;windowfn)

A windowing function, which is a function that can be passed to [`compute_freqbins`](#).
This one selects a window relative to the target, if present. If the target isn't present,
selects a random window.

If passed a start and length, these are relative to the target, in seconds. If passed a
windowfn that function takes a DataFrameRow, containing the event information for a trial,
and it should return a named tuple with `start` and `len`.

"""
const target_seed = 1983_11_09
function windowtarget(;params...)
    WindowFn(:target; params...)
end
function slice(wn::WindowFnBounds{:target}, trial, event, fs)
    time = !ismissing(event.target_time) ? event.target_time : begin
        maxlen = floor(Int, size(trial, 2) / fs)
        rand(trialrng((:windowtarget_missing, target_seed), event),
            Distributions.Uniform(0, maxlen - to))
    end
    windowtrial(trial, fs, time .+ (wn.start, wn.start + wn.len))
end

"""
    window_target_switch(;start,len)
    window_target_switch(;windowfn)

A windowing function, which is a function that can be passed to [`compute_freqbins`](#).
This one selects a window near the switch just before a target; if no target exists
a random switch is selected.
"""
function window_target_switch(;params...)
    WindowFn(:target_switch; params...)
end
function slice(wn::WindowFnBounds{:target_switch}, trial, event, fs)
    from, to = wn.start, wn.start + wn.len
    si = event.sound_index
    stimes = switch_times[si]

    if ismissing(event.target_time)
        options = only_near(stimes, max_trial_length, window = (from, to))
        window = rand(trialrng((:window_target_switch, switch_seed), event), options)

        start = max(1, round(Int, window[1]*fs))
        stop = min(round(Int, window[2]*fs), size(trial, 2))

        view(trial, :, start:stop)
    else
        times = @_ stimes |> sort |> filter(_ < event.target_time, __)
        isempty(times) && return missing
        time = last(times)

        window = only_near(time, max_trial_length, window=(from, to))
        start = max(1, round(Int, window[1]*fs))
        stop = min(round(Int, window[2]*fs), size(trial, 2))

        view(trial, :, start:stop)
    end
end

"""
    windowswitch(;start, len)
    windowswitch(;windowfn)

A windowing function, which is a function that can be passed to [`compute_freqbins`](#).
This one selects a window near a randomly selected switch.

If passed a start and length, these are relative to the target, in seconds. If passed a
windowfn that function takes a DataFrameRow, containing the event information for a trial,
and it should return a named tuple with `start` and `len`.

"""
const switch_seed = 2018_11_18
function windowswitch(;params...)
    WindowFn(:switch; params...)
end
function slice(wn::WindowFnBounds{:switch}, trial, event, fs)
    si = event.sound_index
    stimes = switch_times[si]

    options = windowsat(stimes, max_trial_length, window = (wn.start, wn.start+wn.len))
    from, to = rand(trialrng((:windowswitch, switch_seed), event), options)

    windowtrial(trial, fs, (from, to))
end

"""
    windowbaseline(;start, len, mindist, minlength, onempty = error)
    windowbaseline(;widnowfn, mindist, minlength, onempty = error)

A windowing function, which is a function that can be passed to [`compute_freqbins`](#).
This one selects a random window far from a switch; this is called a baseline, because these
regions are generally used as a control to compare to the target. We avoid switches because
we have a priori reason to think the response might be different during these disorienting
moments.

If passed a start and length, these are relative to the target, in seconds. If passed a
windowfn that function takes a DataFrameRow, containing the event information for a trial,
and it should return a named tuple with `start` and `len`.

## Parameters
- `mindist`: minimum to switch edge (start or end)
- `minlength`: miniumum length of baseline regions considered (shorter ones are filtered
  out)
- `onempty`: what to do when there are no baseline regions, possible values are `missing`
  (return `missing` value), `error` (throw an error) or zero argument function, which will
  be called if the result is empty.

"""
const baseline_seed = 2017_09_16
function windowbaseline(;mindist, minlength, onempty = error, params...)
    WindowFn(:baseline, (mindist, minlength, onempty); params...)
end
function slice(wn::WindowFnBounds{:baseline}, trial, event, fs)
    from, to = wn.start, wn.start + wn.len
    (mindist, minlength, onempty) = wn.entry

    handleempty(onempty::Function) = onempty()
    handleempty(::Missing) = missing
    handleempty(::typeof(error)) =
        error("Could not find any valid region for baseline window.")

    si = event.sound_index
    times = !ismissing(event.target_time) ?
        vcat(switch_times[si], event.target_time) |> sort! :
        switch_times[si]
    ranges = windows_far_from(times, max_trial_length, mindist = mindist, minlength = minlength)
    isempty(ranges) && return handleempty(onempty)

    at = sample_from_ranges(trialrng((:windowbaseline, baseline_seed), event), ranges)
    window = windowsat(at, fs, window = (from, to))

    start = max(1, round(Int, window[1]*fs))
    stop = min(round(Int, window[2]*fs), size(trial, 2))
    view(trial, :, start:stop)
end

# Feature Extraction
# =================================================================

# First, define the frequency bin names and boundaries
const default_freqbins = OrderedDict(
    :delta => (1, 3),
    :theta => (3, 7),
    :alpha => (7, 15),
    :beta => (15, 30),
    :gamma => (30, 100)
)

# Utilities
# -----------------------------------------------------------------

# `windowbounds` converts from start,len to from,to format
windowbounds(x) = (x.start, x.start + x.len)

# Most windows are defined by a start,len named tuple, but some need
# more information about the individual trial to pick the right time;
# these are represented as an anoymouse function which is passed the
# individual time
windowparams(x::NamedTuple,_) = x
windowparams(fn::Function, event) = windowparams(fn(event), event)

# Helper Functions
# -----------------------------------------------------------------

"""
    compute_powerbin_features(eeg, data, windowfn; freqbins = default_freqbins,
        channels = Colon())

For a given subset of data and windowing (defined by `windowfn`) compute a single feature
vector; there are nchannels x nfreqbins total features. The features are computed using
[`computebands`](#).

Features are weighted by the number of observations (valid windows) they represent.
"""
function compute_powerbin_features(eeg, data, windowfn; kwds...)
    @assert data.sid |> unique |> length == 1 "Expected one subject's data"
    sid = data.sid |> first

    # TODO: this code is likely to be type unstable; could gain some speed
    # by optimizing it, but hasn't been worth it to me yet
    fs = framerate(eeg)
    windowfn = resolve(windowfn, data)
    windows = @_ map(windowfn(eeg[_1.trial_index], _1, fs), eachrow(data))
    isempty(skipmissing(windows)) && return Empty(DataFrame)

    signal = reduce(hcat, skipmissing(windows))
    weight = sum(!isempty, skipmissing(windows))
    freqdf = computebands(signal, fs; kwds...)
    freqdf[!, :weight] .= weight

    if size(freqdf, 1) > 0
        nbins = size(freqdf, 2)-2
        powerdf = @_ freqdf |>
            stack(__, 1:nbins,
                variable_name = :freqbin, value_name = :power) |>
            groupby(__, :channel) |>
            transform!(__, :weight => minimum => :weight) |>
            filter(all(!isnan, _.power), __)

        if any(ismissing, Array(powerdf))
            @infiltrate
        end

        chstr = @_(map(@sprintf("%02d", _), powerdf.channel))
        features = Symbol.("channel_", chstr, "_", powerdf.freqbin)
        DataFrame(
            winlen     =  windowfn.len,
            winstart   =  windowfn.start,
            windowtype =  windowfn.name,
            weight     =  minimum(powerdf.weight);
            (features .=> powerdf.power)...
        )
    else
        Empty(DataFrame)
    end
end

"""
    computebands(signal, fs; freqbins = default_freqbins, channels = Colon())

For a given signal (and sample rate `fs`), compute the power in each frequency bin.
"""
function computebands(signal, fs; freqbins = default_freqbins, channels = Colon())

    function freqrange(spect, (from, to))
        freqs = range(0, fs/2, length = size(spect, 2))
        view(spect, :, findall(from-step(freqs)*0.51 .≤ freqs .≤ to+step(freqs)*0.51))
    end

    if size(signal, 2) < 32
        Empty(DataFrame)
    end
    if size(signal, 2) < 128
        newsignal = similar(signal, size(signal, 1), 128)
        newsignal[:, 1:size(signal, 2)] = signal
        newsignal[:, (size(signal, 2)+1):end] .= 0
        signal = newsignal
    end
    spect = abs.(rfft(view(signal, channels, :), 2))
    # totalpower = mean(spect, dims = 2)
    result = mapreduce(hcat, keys(freqbins)) do bin
        mfreq = mean(freqrange(spect, freqbins[bin]), dims = 2) #./ totalpower
        DataFrame(Symbol(bin) => vec(mfreq))
    end
    if channels isa Colon
        result[!, :channel] .= 1:size(result, 1)
    else
        result[!, :channel] .= channels
    end

    if @_ all(0 ≈ _, signal)
        result[!, Symbol.(keys(freqbins))] .= 0.0
    end

    result
end

"""
    compute_freqbins(subjects, groupdf, windows, [reducerfn = foldxt])

Compute features for a given set of subjects, according to the grouped events,
using `windows`. The subjects are the first return value of `load_all_subjects`,
while `groupdf` is a grouped data frame of the subjects events, which are the seconds
return value of `load_all_subjects`. The windows should be created using the windowing
functions (see above).

"""
function compute_freqbins(subjects, groupdf, windows, reducerfn = foldxt;
    kwds...)

    progress = Progress(length(groupdf) * length(windows),
        desc = "Computing frequency bins...")
    function helper(((key, sdf), window))
        # compute features in each window
        result = compute_powerbin_features(subjects[sdf.sid[1]].eeg, sdf,
            window; kwds...)
        if !isempty(result)
            insertcols!(result, 1, (keys(key) .=> values(key))...)
        end

        next!(progress)
        result
    end
    classdf = @_ groupdf |> pairs |> collect |> Iterators.product(__, windows) |>
        reducerfn(append!!, Map(helper), __)
    ProgressMeter.finish!(progress)

    classdf
end

