export compute_powerdiff_features, computebands

function compute_powerdiff_features(eeg,data,region,window)
    fs = framerate(eeg)

    freqdf = mapreduce(append!!,[:before,:after]) do timing
        windows = map(eachrow(data)) do row
            bounds = timing == :before ?
                (window.before, window.before + window.len) :
                (window.start, window.start + window.len)
            region == "target" ?
                windowtarget(eeg[row.trial_index],row,fs,bounds...) :
                windowbaseline(eeg[row.trial_index],row,fs,bounds...)
        end
        signal = reduce(hcat,windows)
        weight = sum(!isempty,windows)
        freqdf = computebands(signal,fs)
        freqdf[!,:window_timing] .= string(timing)
        freqdf[!,:weight] .= weight

        freqdf
    end

    if size(freqdf,1) > 0
        powerdf = @_ freqdf |>
            stack(__, Between(:delta,:gamma),
                variable_name = :freqbin, value_name = :power) |>
            groupby(__,:channel) |>
            transform!(__,:weight => minimum => :weight) |>
            filter(all(!isnan,_.power), __) |>
            unstack(__, :window_timing, :power)

        ε = 1e-8
        logdiff(x,y) = log.(ε .+ x) .- log.(ε .+ y)
        powerdiff = logdiff(powerdf.after,powerdf.before)

        chstr = @_(map(@sprintf("%02d",_),powerdf.channel))
        features = Symbol.("channel_",chstr,"_",powerdf.freqbin)
        DataFrame(weight=minimum(powerdf.weight);(features .=> powerdiff)...)
    else
        Empty(DataFrame)
    end
end

function computebands(signal,fs;freqbins=OrderedDict(
        :delta => (1,3),
        :theta => (3,7),
        :alpha => (7,15),
        :beta => (15,30),
        :gamma => (30,100)))

    function freqrange(spect,(from,to))
        freqs = range(0,fs/2,length=size(spect,2))
        view(spect,:,findall(from-step(freqs)*0.51 .≤ freqs .≤ to+step(freqs)*0.51))
    end

    if size(signal,2) < 32
        Empty(DataFrame)
    end
    if size(signal,2) < 128
        newsignal = similar(signal,size(signal,1),128)
        newsignal[:,1:size(signal,2)] = signal
        newsignal[:,(size(signal,2)+1):end] .= 0
        signal = newsignal
    end
    spect = abs.(rfft(signal, 2))
    # totalpower = mean(spect,dims = 2)
    result = mapreduce(hcat,keys(freqbins)) do bin
        mfreq = mean(freqrange(spect, freqbins[bin]), dims = 2) #./ totalpower
        DataFrame(Symbol(bin) => vec(mfreq))
    end
    result[!,:channel] .= 1:size(result,1)

    if @_ all(0 ≈ _,signal)
        result[!,Between(:delta,:gamma)] .= 0.0
    end

    result
end
