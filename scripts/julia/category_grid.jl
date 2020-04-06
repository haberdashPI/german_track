using DrWatson
@quickactivate("german_track")

using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures,
    Dates, Underscores, StatsBase, Random, Printf, ProgressMeter

import GermanTrack: stim_info, speakers, directions, target_times, switch_times

eeg_files = dfhit = @_ readdir(data_dir()) |> filter(occursin(r".mcca$",_), __)

import GermanTrack: stim_info, speakers, directions, target_times, switch_times
subjects = Dict(file => load_subject(joinpath(data_dir(), file), stim_info,
                                     encoding = RawEncoding())
    for file in eeg_files)

target_salience =
    CSV.read(joinpath(stimulus_dir(), "target_salience.csv")).salience |> Array
med_salience = median(target_salience)

regions = ["target", "baseline"]
timings = ["before", "after"]
source_names = ["male", "female"]
winlens = range(0,2,length=10)
winstart = range(0,2,length=10)
fs = GermanTrack.framerate(first(values(subjects)).eeg)

get_target_source(row) = get(source_names,Int.(row.target_source),missing)

event_cols = [:correct,:target_present,:target_source,:condition,:trial,
    :sound_index,:target_time]

N = reduce(*,length.((values(subjects),regions,timings,winlens,winstart)))
progres = Progress(N,desc="computing frequency bins")
df = mapreduce(vcat,values(subjects)) do subject
    events = subject.events
    eeg = subject.eeg

    rows = @_ 1:size(events,1) |>
        filter(events[_,:target_present] == 1,__) |>
        filter("hit" == ishit(events[_,:],get_target_source(events[_,:]),
                             "target",directions[events[_,:sound_index]]),__)

    mapreduce(hcat,Iterators.product(regions,timings,winlens,winstart)) do vars
        region,timing,winlen,winstart = vars

        bounds = timing == "before" ? (-winstart-winlen,-winstart) :
                (winstart,winstart+winlen)

        # TODO: group rows by the categories below (salience direct etc...
        # and hit and use that to compute the signal)
        signal = mapreduce(hcat,rows) do row
            region == "target" ?
                windowtarget(eeg[row],events[row,:],fs,bounds...) :
                windowbaseline(eeg[row],events[row,:],fs,mindist=0.2,minlen=0.5)
        end

        freqbands = computebands(signal,fs)

        freqbands[:,:salience] = target_salience[si] > med_salience ? "high" : "low"
        freqbands[:,:target_time] = target_times[si] > med_target_time ? "early" : "late"
        freqbands[:,:direction] = directions[si]

        next!(progres)
        freqbands
    end
end

cols = [:sid,:hit,:trial,:timing,:condition,:winstart,:winlen,:salience]
N = length(groupby(dfhit,cols))
progress = Progress(N, "Computing Frequency Bins: ")
freqmeans = by(dfhit, cols) do rows
    # @assert size(rows,1) == 1
    # signal = rows.eeg[1]
    signal = reduce(hcat,row.eeg for row in eachrow(rows))
    # ensure a minimum of 2Hz freqbin resolution
    if size(signal,2) < 32
        empty = mapreduce(hcat,keys(freqbins)) do bin
            DataFrame(Symbol(bin) => Float64[])
        end
        empty[!,:channel] = Int[]
        next!(progress)
        return empty
    end
    if size(signal,2) < 128
        newsignal = similar(signal,size(signal,1),128)
        newsignal[:,1:size(signal,2)] = signal
        signal = newsignal
    end
    spect = abs.(rfft(signal, 2))
    # totalpower = mean(spect,dims = 2)
    result = mapreduce(hcat,keys(freqbins)) do bin
        mfreq = mean(freqrange(spect, freqbins[bin]), dims = 2) #./ totalpower
        DataFrame(Symbol(bin) => vec(mfreq))
    end
    result[!,:channel] .= channels
    next!(progress)
    result
end
