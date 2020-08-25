export compute_powerdiff_features, compute_powerbin_features, computebands,
    windowtarget, windowbaseline, windowswitch, compute_freqbins

using Random123 # counter-based random number generators, this lets use reliably map
# trial and subject id's to a random sequence

only_near(time::Number,max_time;kwds...) =
    only_near((time,),max_time;kwds...)[1]
function only_near(times,max_time;window=(-0.250,0.250))
    result = Array{Tuple{Float64,Float64}}(undef,length(times))

    i = 0
    stop = 0
    for time in times
        new_stop = min(time+window[2],max_time)
        if stop < time+window[1]
            i = i+1
            result[i] = (time+window[1],new_stop)
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

function not_near(times,max_time;window=(0,0.5))
    result = Array{Tuple{Float64,Float64}}(undef,length(times)+1)

    start = 0
    i = 0
    for time in times
        if start < time
            i = i+1
            result[i] = (start,time+window[1])
        end
        start = time+window[2]
    end
    if start < max_time
        i = i+1
        result[i] = (start,max_time)
    end

    view(result,1:i)
end

function far_from(times,max_time;mindist=0.5,minlength=0.5)
    result = Array{Tuple{Float64,Float64}}(undef,length(times)+1)
    start = 0
    i = 0
    for time in times
        if start < time
            if time-mindist-minlength > start
                i = i+1
                result[i] = (start,time-mindist)
            end
            start = time + mindist
        end
    end
    if start < max_time
        i = i+1
        result[i] = (start,max_time)
    end
    view(result,1:i)
end

sample_from_ranges(ranges) = sample_from_ranges(Random.GLOBAL_RNG,ranges)
function sample_from_ranges(rng,ranges)
    weights = Weights(map(x -> x[2]-x[1],ranges))
    range = StatsBase.sample(rng,ranges,weights)
    rand(Distributions.Uniform(range...))
end


const target_seed = 1983_11_09
function windowtarget(trial,event,fs,(from,to))

    time = !ismissing(event.target_time) ? event.target_time : begin
        maxlen = floor(Int, size(trial,2) / fs)
        rand(trialrng((:windowtarget_missing, target_seed), event),
            Distributions.Uniform(0,maxlen - to))
    end

    window = only_near(time, max_trial_length, window=(from, to))
    start = max(1, round(Int, window[1]*fs))
    stop = min(round(Int, window[2]*fs), size(trial, 2))
    view(trial, :, start:stop)
end

const switch_seed = 2018_11_18
function windowswitch(trial, event, fs, (from, to))
    si = event.sound_index
    stimes = switch_times[si]

    options = only_near(stimes, max_trial_length, window = (from, to))
    window = rand(trialrng((:windowswitch, switch_seed), event), options)

    start = max(1, round(Int, window[1]*fs))
    stop = min(round(Int, window[2]*fs), size(trial, 2))
    view(trial, :, start:stop)
end
const NUM_STIMULI = 50
const COND_ID = Dict(
    "global"  => 0,
    "object"  => 1,
    "spatial" => 2,
)
# the random value is determined by the stimulus and condition:
# if it is the same stimulus and condition, the same window is used
function trialrng(id, event)
    seed = stablehash(id)
    Threefry2x((seed, NUM_STIMULI*COND_ID[event.condition] + event.sound_index))
end

const baseline_seed = 2017_09_16
function windowbaseline(;mindist, minlength, onempty = error)
    handleempty(onempty::Missing) = missing
    handleempty(onempty::Function) = onempty()
    handleempty(onempty::typeof(error)) =
        error("Could not find any valid region for baseline window.")

    function(trial, event, fs, (from, to))
        si = event.sound_index
        times = target_times[si] ≥ 0 ? vcat(switch_times[si], target_times[si]) |> sort! :
            switch_times[si]
        ranges = far_from(times, max_trial_length, mindist = mindist, minlength = minlength)
        isempty(ranges) && return handleempty(onempty)

        at = sample_from_ranges(trialrng((:windowbaseline, baseline_seed), event), ranges)
        window = only_near(at, fs, window = (from, to))

        start = max(1, round(Int, window[1]*fs))
        stop = min(round(Int, window[2]*fs), size(trial, 2))
        view(trial, :, start:stop)
    end
end

const default_freqbins = OrderedDict(
    :delta => (1, 3),
    :theta => (3, 7),
    :alpha => (7, 15),
    :beta => (15, 30),
    :gamma => (30, 100)
)

windowbounds(x::NamedTuple,_) = (x.start, x.start + x.len)
windowbounds(fn::Function, event) = windowbounds(fn(event), event)

windowbounds(x::NamedTuple,_) = (x.start, x.start + x.len)
windowbounds(fn::Function, event) = windowbounds(fn(event), event)

function compute_powerbin_features(eeg, data, windowfn, window; kwds...)
    fs = framerate(eeg)
    @infiltrate
    windows = @_ map(windowfn(eeg[_1.trial_index], _1, fs,
        windowbounds(window,_1)), eachrow(data))
    signal = reduce(hcat, skipmissing(windows))
    weight = sum(!isempty, skipmissing(windows))
    freqdf = computebands(signal, fs; kwds...)
    freqdf[!, :weight] .= weight

    freqdf

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
        DataFrame(weight = minimum(powerdf.weight);(features .=> powerdf.power)...)
    else
        Empty(DataFrame)
    end
end

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

function compute_freqbins(subjects, groupdf, windowfn, windows, reducerfn = foldxt;
    kwds...)

    progress = Progress(length(groupdf), desc = "Computing frequency bins...")
    classdf = @_ groupdf |>
        combine(function(sdf)
            # compute features in each window
            function findwindows(window)
                result = compute_powerbin_features(subjects[sdf.sid[1]].eeg, sdf,
                    windowfn, window; kwds...)
                result[!, :winstart] .= window.start
                result[!, :winlen] .= window.len
                result
            end
            x = reducerfn(append!!, Map(findwindows), windows)
            next!(progress)
            x
        end, __)
    ProgressMeter.finish!(progress)

    classdf
end

