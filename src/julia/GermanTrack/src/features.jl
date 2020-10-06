export windowtarget, windowbaseline, windowswitch, compute_freqbins

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

struct WindowFn{S}
    start::Float64
    len::Float64
end

"""
    windowtarget((;start, len))
    windowtarget(fn)

A windowing function, which is a function that can be passed to [`compute_freqbins`](#).
This one selects a window relative to the target, if present. If the target isn't present,
selects a random window.

If passed a start and length, these are relative to the target, in seconds. If passed a
function that function takes a DataFrameRow, containing the event information for a trial,
and it should return a named tuple with `start` and `len`.

"""
const target_seed = 1983_11_09
function windowtarget(params)
    WindowFn{:target}(windowparams(params)...)
end
function slice(wn::WindowFn{:target}, trial, event, fs)
    time = !ismissing(event.target_time) ? event.target_time : begin
        maxlen = floor(Int, size(trial, 2) / fs)
        rand(trialrng((:windowtarget_missing, target_seed), event),
            Distributions.Uniform(0, maxlen - to))
    end
    windowtrial(trial, fs, time .+ (wn.start, wn.start + wn.len))
end

"""
    windowswitch(trial, event, fs, (from, to))

A windowing function, which is a function that can be passed to [`compute_freqbins`](#).
This one selects a window near a randomly selected switch.

"""
const switch_seed = 2018_11_18
function windowswitch(params)
    WindowFn{:switch}(windowparams(params)...)
function slice(wn::WindowFn{:switch}, trial, event, fs)
    from, to = start, start+len
    function (trial, event, fs)
        si = event.sound_index
        stimes = switch_times[si]

        options = windowsat(stimes, max_trial_length, window = (from, to))
        from, to = rand(trialrng((:windowswitch, switch_seed), event), options)

        windowtrial(trial, fs, (from, to))
    end
end

"""
    windowbaseline(trial, event, fs, (from, to))

A windowing function, which is a function that can be passed to [`compute_freqbins`](#).
This one selects a random window far from a switch; this is called a baseline, because these
regions are generally used as a control to compare to the target. We avoid switches because
we have a priori reason to think the response might be different during these disorienting
moments.

"""
const baseline_seed = 2017_09_16
function windowbaseline(;start, len, mindist, minlength, onempty = error)
    from, to = start, start + len

    handleempty(onempty::Missing) = missing
    handleempty(onempty::Function) = onempty()
    handleempty(onempty::typeof(error)) =
        error("Could not find any valid region for baseline window.")

    function(trial, event, fs, (from, to))
        si = event.sound_index
        times = target_times[si] ≥ 0 ? vcat(switch_times[si], target_times[si]) |> sort! :
            switch_times[si]
        ranges = windows_far_from(times, max_trial_length, mindist = mindist, minlength = minlength)
        isempty(ranges) && return handleempty(onempty)

        at = sample_from_ranges(trialrng((:windowbaseline, baseline_seed), event), ranges)
        window = windowsat(at, fs, window = (from, to))

        start = max(1, round(Int, window[1]*fs))
        stop = min(round(Int, window[2]*fs), size(trial, 2))
        view(trial, :, start:stop)
    end
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
    compute_powerbin_features(eeg, data, windowfn, window; freqbins = default_freqbins,
        channels = Colon())

For a given subset of data and windowing (defined by `windowfn` and `window`) compute a
single feature vector; there are nchannels x nfreqbins total features. The
features are computed using [`computebands`](#).

Features are weighted by the number of observations (valid windows) they represent.
"""
function compute_powerbin_features(eeg, data, windowfn, window; kwds...)
    @assert data.sid |> unique |> length == 1 "Expected one subject's data"
    sid = data.sid |> first

    fs = framerate(eeg)
    wparams = windowparams(window, sid)
    windows = @_ map(windowfn(eeg[_1.trial_index], _1, fs, windowbounds(wparams)),
        eachrow(data))
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

        chstr = @_(map(@sprintf("%02d", _), powerdf.channel))
        features = Symbol.("channel_", chstr, "_", powerdf.freqbin)
        DataFrame(
            winlen     =  wparams.len,
            winstart   =  wparams.start,
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
    compute_freqbins(subjects, groupdf, windowfn, window, [reducerfn = foldxt])
"""
function compute_freqbins(subjects, groupdf, windowfn, windows, reducerfn = foldxt;
    kwds...)

    progress = Progress(length(groupdf), desc = "Computing frequency bins...")
    classdf = @_ groupdf |>
        combine(function(sdf)
            # compute features in each window
            function findwindows(window)
                result = compute_powerbin_features(subjects[sdf.sid[1]].eeg, sdf,
                    windowfn, window; kwds...)
                result
            end
            x = reducerfn(append!!, Map(findwindows), windows)
            next!(progress)
            x
        end, __)
    ProgressMeter.finish!(progress)

    classdf
end

