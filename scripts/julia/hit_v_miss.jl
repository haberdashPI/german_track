using DrWatson
@quickactivate("german_track")

using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures, FFTW,
    Dates, LIBSVM, Underscores

# eeg_files = filter(x->occursin(r"_mcca03\.mcca_proj$", x), readdir(data_dir()))
eeg_files = filter(x->occursin(r".mcca$", x), readdir(data_dir()))
# eeg_files = filter(x->occursin(r"_cleaned\.eeg$", x), readdir(data_dir()))
eeg_encoding = RawEncoding()

import GermanTrack: stim_info, speakers, directions, target_times, switch_times
subjects = Dict(file => load_subject(joinpath(data_dir(), file), stim_info,
                                     encoding = eeg_encoding)
    for file in eeg_files)

target_salience =
    CSV.read(joinpath(stimulus_dir(), "target_salience.csv")).salience |> Array
med_salience = median(target_salience)

const tindex = Dict("male" => 1, "fem" => 2)

halfway = @_ filter(_ > 0,target_times) |> median
regions = ["target", "baseline"]
timings = ["before", "after"]
winlens = (0.5,1,1.5)
winstarts = (0.0,0.25,0.5)
fs = GermanTrack.framerate(first(values(subjects)).eeg)
factors = Iterators.product(regions,timings,winlens,winstarts)

df = mapreduce(vcat,values(subjects)) do subject
    rows = filter(1:size(subject.events,1)) do i
        !subject.events.bad_trial[i] && subject.events.target_present[i] == 1
    end

    mapreduce(vcat,rows) do row
        si = subject.events.sound_index[row]
        event = subject.events[row,[:correct,:target_present,:target_source,
            :condition,:trial,:sound_index,:target_time]] |> copy

        mapreduce(vcat,factors) do (region,timing,len,start)
            winbounds = timing == "before" ? (-start-len,-start) :
                (start,start+len)
            window = if region == "target"
                only_near(event.target_time,fs,window=winbounds)
            else
                times = vcat(switch_times[si], target_times[si]) |> sort!
                ranges = far_from(times, 10, mindist=0.2, minlength=0.5)
                if isempty(ranges)
                    error("Could not find any valid region for baseline ",
                          "'target'. Times: $(times)")
                end
                at = sample_from_ranges(ranges)
                only_near(at,fs,window=winbounds)
            end

            maxlen = size(subject.eeg[row],2)
            ixs = bound_indices(window,fs,maxlen)
            maxtime = maxlen*fs
            DataFrame(;
                event...,
                region = region,
                timing = timing,
                winlen = len,
                winstart = start,
                sid = subject.sid,
                direction = directions[si],
                salience = target_salience[si] > med_salience ? "high" : "low",
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

df.hit = ishit.(eachrow(df))
dfhit = df[in.(df.hit,Ref(("hit","miss","baseline"))),:]

freqbins = OrderedDict(
    :delta => (1,3),
    :theta => (3,7),
    :alpha => (7,15),
    :beta => (15,30),
    :gamma => (30,100),
)

fs = GermanTrack.framerate(first(values(subjects)).eeg)
# channels = first(values(subjects)).eeg.label
channels = 1:30
function freqrange(spect,(from,to))
    freqs = range(0,fs/2,length=size(spect,2))
    view(spect,:,findall(from-step(freqs)*0.51 .≤ freqs .≤ to+step(freqs)*0.51))
end

freqmeans = by(dfhit, [:sid,:hit,:timing,:condition,:winstart,:winlen,:salience]) do rows
    # @assert size(rows,1) == 1
    # signal = rows.eeg[1]
    signal = reduce(hcat,row.eeg for row in eachrow(rows))
    if size(signal,2) < 100
        empty = mapreduce(hcat,keys(freqbins)) do bin
            DataFrame(Symbol(bin) => Float64[])
        end
        empty[!,:channel] = Int[]
        return empty
    end
    spect = abs.(rfft(signal, 2))
    # totalpower = mean(spect,dims = 2)
    result = mapreduce(hcat,keys(freqbins)) do bin
        mfreq = mean(freqrange(spect, freqbins[bin]), dims = 2) #./ totalpower
        DataFrame(Symbol(bin) => vec(mfreq))
    end
    result[!,:channel] .= channels
    result
end

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

R"""

library(ggplot2)
library(dplyr)
library(tidyr)

bins = $(collect(keys(freqbins)))

# TODO: show the varying window parameters within a single graph
# so I can understand how things change,
# also, think about the comparisons I care about
# and other ways to collapse across more of the figures (e.g. colors
# for the different frequency bands, or something)

df = $(freqmeans) %>%
    # filter(channel %in% 1:3) %>%
    group_by(sid,winstart,winlen,hit,#target_timing, salience,
        timing,condition) %>%
    gather(key="freqbin", value="meanpower", delta:gamma) %>%
    ungroup() %>%
    mutate(timing = factor(timing,levels=c('before','after')),
           freqbin = factor(freqbin,levels=bins, ordered=T),
           condition = factor(condition,levels=c('global','object','spatial'))) %>%
    arrange(timing,freqbin)

group_means = df %>%
    group_by(sid,winstart,winlen,hit,#target_timing, salience,
        timing,condition,freqbin) %>%
    summarize(meanpower = median(meanpower))

# for(start in unique(df$winstart)){
#     for(len in unique(df$winlen)){
plotdf = group_means %>%
    # filter(group_means,freqbin %in% c('delta','theta','alpha')) %>%
    # filter(condition %in% c('global','object')) %>%
    #filter(group_means,winstart == start,winlen == len) %>%
    filter(hit %in% c('hit','miss')) %>%
    group_by(sid,hit,condition) %>%
    spread(timing,meanpower) %>%
    mutate(diff = log(after) - log(before))

pos = position_jitterdodge(dodge.width=0.1,jitter.width=0.05)

p = ggplot(plotdf,aes(x=winstart,y=diff,
    fill=interaction(hit,factor(winlen)),
    color=interaction(hit,factor(winlen)))) +
    stat_summary(geom='line',position=position_dodge(width=0.1),
        size=1) +
    stat_summary(geom='linerange',position=position_dodge(width=0.1),
                 fun.args = list(conf.int=0.68)) +
    # geom_point(alpha=0.1, position=pos) +
    scale_fill_brewer(palette='Paired',direction=-1) +
    scale_color_brewer(palette='Paired',direction=-1) +
    facet_grid(freqbin~condition,scales='free_y') +
    ylab("Median log power difference across channels (after - before)") +
    # coord_cartesian(ylim=c(-0.1,0.1)) +
    geom_abline(slope=0,intercept=0,linetype=2)
p

# name = sprintf('freq_diff_summary_target_timing_%03.1f_%03.1f.pdf',start,len)
# name = sprintf('mcca_freq_diff_summary_target_timing_%03.1f_%03.1f.pdf',start,len)
name = 'hits_by_all_windows.pdf'
ggsave(file.path($dir,name),plot=p,width=11,height=8)
#     }
# }

plotdf = group_means %>%
    filter(((winstart == 0.25) & (winlen == 0.5)) |
           ((winstart == 0.5) & (winlen == 1.5))) %>%
    group_by(sid,hit,condition) %>%
    spread(timing,meanpower) %>%
    mutate(diff = log(after) - log(before))

pos = position_jitterdodge(dodge.width=0.2,jitter.width=0.1)

p = ggplot(plotdf,aes(x=freqbin,y=diff,
        fill=hit,color=hit)) +
    # geom_point(alpha=0.4, position=pos, size=1) +
    stat_summary(fun.data = "mean_cl_boot", geom='point',
        position=position_dodge(width=0.4),
        size=2,fun.args = list(conf.int=0.68)) +
    stat_summary(fun.data = "mean_cl_boot", geom='errorbar',
        position=position_dodge(width=0.4), width=0.5,
        fun.args = list(conf.int=0.68)) +
    # geom_text(position=pos, size=4, aes(label=sid)) +
    scale_fill_brewer(palette='Set1',direction=-1) +
    scale_color_brewer(palette='Set1',direction=-1) +
    # coord_cartesian(ylim=c(-0.01,0.01)) +
    facet_grid(.~condition+winlen,scales='free_y') +
    theme(axis.text.x = element_text(angle=90,hjust=1)) +
    ylab("Median log power difference across channels (after - before)") +
    geom_abline(slope=0,intercept=0,linetype=2)
p

name = sprintf('hits_by_select_windows.pdf',0.5,1.0)
ggsave(file.path($dir,name),plot=p,width=11,height=8)

df = $(freqmeans) %>%
    # filter(channel %in% 1:3) %>%
    mutate(timing = factor(timing,levels=c('before','after')),
           condition = factor(condition,levels=c('global','object','spatial'))) %>%
    arrange(timing,condition)

plotdf = df %>%
    filter(((winstart == 0.25) & (winlen == 0.5)) |
           ((winstart == 0.5) & (winlen == 1.5))) %>%
    group_by(sid,hit,condition,channel,winstart,winlen) %>%
    select(timing,alpha) %>%
    spread(timing,alpha) %>%
    summarize(meandiff = mean(log(after) - log(before)))

p = ggplot(filter(plotdf,channel <= 5),aes(x=channel,y=meandiff,
        fill=hit,color=hit)) +
    # geom_point(alpha=0.4, position=pos, size=1) +
    stat_summary(fun.data = "mean_cl_boot", geom='point',
        position=position_dodge(width=0.4),
        size=2,fun.args = list(conf.int=0.68)) +
    stat_summary(fun.data = "mean_cl_boot", geom='errorbar',
        position=position_dodge(width=0.4), width=0.5,
        fun.args = list(conf.int=0.68)) +
    # geom_text(position=pos, size=4, aes(label=sid)) +
    scale_fill_brewer(palette='Set1',direction=-1) +
    scale_color_brewer(palette='Set1',direction=-1) +
    # coord_cartesian(ylim=c(-0.01,0.01)) +
    facet_grid(.~condition+winlen,scales='free_y') +
    ylab("Median power difference across channels (after - before)") +
    xlab("MCCA Component")
    geom_abline(slope=0,intercept=0,linetype=2)
p

name = sprintf('hits_alpha_by_channel_select_windows.pdf',0.5,1.0)
ggsave(file.path($dir,name),plot=p,width=11,height=8)

"""

svmdf = @_ freqmeans |>
    filter((_1.winstart == 0.25 && _1.winlen == 0.5) ||
           (_1.winstart == 0.5 && _1.winlen == 1.5),__) |>
    filter(_.condition in ["global","object"],__) |>
    stack(__, [:delta,:theta,:alpha,:beta,:gamma],
        variable_name = :freqbin, value_name = :power) |>
    unstack(__, :timing, :power) |>
    by(__, [:sid,:freqbin,:condition,:winstart,:winlen,:channel,:salience],
        (:before,:after) => sdf ->
            (powerdiff = mean(log.(sdf.after) .- log.(sdf.before)),))

svmclass = by(svmdf,[:winstart,:winlen,:channel,:salience,:freqbin]) do sdf
    N = 0
    correct = 0
    for (train_ids,test_ids) in folds(10,unique(sdf.sid))
        train = @_ filter(_.sid in train_ids,sdf)
        test = @_ filter(_.sid in test_ids,sdf)
        model = svmtrain(Array(train[:,[:powerdiff]])',
                         Array(train[:,:condition]))
        labels, = svmpredict(model, Array(test[:, [:powerdiff]])')
        N += size(test,1)
        correct += sum(labels .== test[:, :condition])
    end
    DataFrame(N=N,correct=correct)
end
svmclass.freqbin = String.(svmclass.freqbin)

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

R"""

library(ggplot2)
library(dplyr)
library(tidyr)

bins = $(collect(keys(freqbins)))

plotdf = $svmclass %>%
    mutate(freqbin = factor(freqbin,levels=bins, ordered=T)) %>%
    arrange(freqbin)

p = ggplot(plotdf,aes(x=channel,y=freqbin,fill=correct/N)) +
    geom_raster() + facet_grid(winstart~salience,labeller="label_both") +
    scale_fill_distiller(name="Label Accuracy (global v object)",
        type="div",palette=5,limits=c(0,1),direction=0) +
    xlab('MCCA Component')
p

ggsave(file.path($dir,"svm_freqbin_salience.pdf"),plot=p,width=11,height=8)

"""
