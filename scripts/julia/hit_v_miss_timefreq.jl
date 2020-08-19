using DrWatson
@quickactivate("german_track")
using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures, FFTW,
    Dates, ProgressMeter, DSP, Underscores, PaddedViews

# eeg_files = filter(x->occursin(r"_mcca03\.mcca_proj$", x), readdir(processed_datadir("eeg")))
eeg_files = filter(x->occursin(r".mcca$", x), readdir(processed_datadir("eeg")))
# eeg_files = filter(x->occursin(r"_cleaned\.eeg$", x), readdir(processed_datadir("eeg")))
eeg_encoding = RawEncoding()

import GermanTrack: stim_info, speakers, directions, target_times, switch_times
subjects = Dict(file => load_subject(joinpath(processed_datadir("eeg"), file), stim_info,
                                     encoding = eeg_encoding)
    for file in eeg_files)

const tindex = Dict("male" => 1, "fem" => 2)

halfway = @_ filter(_ > 0,target_times) |> median
regions = ["target", "baseline"]
fs = GermanTrack.framerate(first(values(subjects)).eeg)

width = 3

df = mapreduce(vcat,values(subjects)) do subject
    rows = filter(1:size(subject.events,1)) do i
        !subject.events.bad_trial[i] && subject.events.target_present[i] == 1
    end

    mapreduce(vcat,rows) do row
        si = subject.events.sound_index[row]
        event = subject.events[row,[:target_detected,:target_present,:target_source,
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
channels = 1:30

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

progres = Progress(length(unique(dfhit.condition))*length(unique(dfhit.sid)))
medpower = by(freqpower, [:condition,:sid]) do sdf
    hit = by(view(sdf,sdf.hit .== "hit",:),[:freq,:time],:power => median)
    miss = by(view(sdf,sdf.hit .== "miss",:),[:freq,:time],:power => median)
    base = by(view(sdf,sdf.hit .== "baseline",:),[:freq,:time],:power => median)

    size(hit,1) == 0 && (hit[!,:power_median] = Float64[])
    size(miss,1) == 0 && (miss[!,:power_median] = Float64[])
    size(base,1) == 0 && (base[!,:power_median] = Float64[])

    rename!(hit,Dict(:power_median => "hit"))
    rename!(miss,Dict(:power_median => "miss"))
    rename!(base,Dict(:power_median => "base"))

    hitvbase = join(hit,base,on=[:freq,:time],kind=:outer)
    hitvbase[!,:hitvbase] = log.(hitvbase.hit) .- log.(hitvbase.base)
    missvbase = join(miss,base,on=[:freq,:time],kind=:outer)
    missvbase[!,:missvbase] = log.(missvbase.miss) .- log.(missvbase.base)

    next!(progress)

    join(missvbase[:,[:time,:freq,:miss,:missvbase]],
         hitvbase[:,[:time,:freq,:hit,:hitvbase]],on=[:time,:freq],kind=:outer)
end

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

R"""

library(ggplot2)
library(dplyr)
library(tidyr)

dfplot = $medpower %>%
    group_by(condition,sid,time,freq) %>%
    summarize(logdiff = log(hit) - log(miss))

p = ggplot(dfplot,aes(x=time,y=freq,fill=logdiff)) +
    facet_grid(.~condition) + geom_raster() +
    scale_y_continuous(breaks = seq(0,128,by=16)) +
    scale_fill_distiller(type="div",limits=c(-1,1),name='log(hit) - log(miss)') +
    xlim(-1,3)
p

ggsave(file.path($dir,"timefreq_hits.pdf"),width=11,height=8)

dfplot2 = dfplot %>%
    mutate(band = ifelse(freq < 9,"delta-theta",
                  ifelse(freq > 30 & freq < 50, "low-gamma", "other"))) %>%
    filter(band != "other") %>%
    group_by(condition,sid,time,band) %>%
    summarize(logdiff = mean(logdiff))

p = ggplot(dfplot2,aes(x=time,y=logdiff,color=condition,fill=condition)) +
    geom_abline(slope=0,intercept=0,linetype=2) +
    stat_summary(fun.data = "mean_cl_boot", geom='line') +
    stat_summary(fun.data = "mean_cl_boot", geom='ribbon',color=NA,alpha=0.5,
                 fun.args = list(conf.int=0.68)) +
    scale_color_brewer(palette='Set1',direction=-1) +
    scale_fill_brewer(palette='Set1',direction=-1) +
    facet_grid(band~.) +
    ylab('log(hit) - log(miss)') +
    xlim(-1,3) +
    coord_cartesian(ylim=c(-1,1))
p

ggsave(file.path($dir,"continuous_band_hits.pdf"),width=11,height=8)

"""
