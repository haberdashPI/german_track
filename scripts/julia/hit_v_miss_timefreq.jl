using DrWatson; @quickactivate("german_track")
include(joinpath(srcdir(), "julia", "setup.jl"))

# eeg_files = filter(x->occursin(r"_mcca03\.mcca_proj$", x), readdir(data_dir()))
eeg_files = filter(x->occursin(r"_mcca34\.mcca_proj$", x), readdir(data_dir()))
# eeg_files = filter(x->occursin(r"_cleaned\.eeg$", x), readdir(data_dir()))
eeg_encoding = RawEncoding()

import GermanTrack: stim_info, speakers, directions, target_times, switch_times
subjects = Dict(file => load_subject(joinpath(data_dir(), file), stim_info,
                                     encoding = eeg_encoding)
    for file in eeg_files)

const tindex = Dict("male" => 1, "fem" => 2)

halfway = filter(@λ(_ > 0),target_times) |> median
regions = ["target", "baseline"]
fs = GermanTrack.framerate(first(values(subjects)).eeg)

width = 3

df = mapreduce(vcat,values(subjects)) do subject
    rows = filter(1:size(subject.events,1)) do i
        !subject.events.bad_trial[i] && subject.events.target_present[i] == 1
    end

    mapreduce(vcat,rows) do row
        si = subject.events.sound_index[row]
        event = subject.events[row,[:correct,:target_present,:target_source,
            :condition,:trial,:sound_index,:target_time]] |> copy

        mapreduce(vcat,regions) do region
            center, window = if region == "target"
                event.target_time, only_near(event.target_time,fs,window=(-width,width))
            else
                times = vcat(switch_times[si], target_times[si]) |> sort!
                ranges = far_from(times, 10, mindist=0.2, minlength=0.5)
                if isempty(ranges)
                    error("Could not find any valid region for baseline ",
                          "'target'. Times: $(times)")
                end
                at = sample_from_ranges(ranges)
                at, only_near(at,fs,window=(-width,width))
            end

            maxlen = size(subject.eeg[row],2)
            ixs = bound_indices(window,fs,maxlen)
            maxtime = maxlen*fs
            DataFrame(;
                event...,
                region = region,
                window_offset = max(0,window[1]) - center,
                sid = subject.sid,
                direction = directions[si],
                eeg = [view(subject.eeg[row],:,ixs)],
            )
        end
    end
end
source_names = ["male", "female"]
df.target_source = get.(Ref(source_names),Int.(df.target_source),missing)

function ishit(row)
    if row.condition == "global"
        row.region == "baseline" ? "baseline" :
            row.correct ? "hit" : "miss"
    elseif row.condition == "object"
        row.region == "baseline" ? "baseline" :
            row.target_source == "male" ?
                (row.correct ? "hit" : "miss") :
                (row.correct ? "falsep" : "reject")
    else
        @assert row.condition == "spatial"
        row.region == "baseline" ? "baseline" :
            row.direction == "right" ?
                (row.correct ? "hit" : "miss") :
                (row.correct ? "falsep" : "reject")
    end
end

df[!,:hit] = ishit.(eachrow(df))
dfhit = df[in.(df.hit,Ref(("hit","miss","baseline"))),:]


# channels = first(values(subjects)).eeg.label
channels = 1:34

using DSP: Periodograms
cols = [:sid,:trial,:hit,:region,:condition]
progress = Progress(length(groupby(dfhit,cols)), "Computing Spectrograms: ")
freqpower = by(dfhit, cols) do rows
    @assert size(rows,1) == 1
    times = nothing
    freqs = nothing
    spects = mapreduce(vcat,Base.axes(rows.eeg[1],1)) do ch
        spect = spectrogram(rows.eeg[1][ch,:], 32, fs=fs, window=DSP.Windows.hanning)
        times = isnothing(times) ? spect.time : times
        freqs = isnothing(freqs) ? spect.freq : freqs
        reshape(spect.power,1,size(spect.power)...)
    end
    next!(progress)
    # aligns window times across conditions
    times = times .+ step(times) * div(rows.window_offset[1],step(times))
    DataFrame(
        power = vec(spects),
        channel = vec(getindex.(CartesianIndices(spects),1)),
        freq = freqs[vec(getindex.(CartesianIndices(spects),2))],
        time = times[vec(getindex.(CartesianIndices(spects),3))]
    )
end

medpower = by(freqpower,
    [:sid,:hit,:region,:condition,:time,:freq],
    (:power,) => x -> (logmed_power = median(log.(x.power)),))
# To start, let's just look at the median TF plot

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

R"""

library(ggplot2)
library(dplyr)
library(tidyr)

p = ggplot($medpower,aes(x=time,y=freq,fill=logmed_power)) +
    facet_grid(condition~region+hit) + geom_raster()
p

ggsave(file.path($dir,"timefreq_hits.pdf"),width=11,height=8)
