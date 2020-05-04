using DrWatson
@quickactivate("german_track")

using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures,
    Dates, Underscores, StatsBase, Random, Printf, ProgressMeter, VegaLite,
    FileIO, StatsBase, LIBSVM

using ScikitLearn
@sk_import svm: (NuSVC, SVC)

import GermanTrack: stim_info, speakers, directions, target_times, switch_times

eeg_files = dfhit = @_ readdir(data_dir()) |> filter(occursin(r".mcca$",_), __)
subjects = Dict(file => load_subject(joinpath(data_dir(), file), stim_info,
                                     encoding = RawEncoding())
    for file in eeg_files)

cachefile = joinpath(cache_dir(),"data","freqmeans.bson")
if !isfile(cachefile)
    freqmeans = organize_data_by(
        subjects,groups=[:salience],hittypes = [:hit,:miss,:baseline],
        winlens = 2.0 .^ range(-3,1,length=10),
        winstarts = 2.0 .^ range(-3,1,length=10)) do signal,fs
            result = computebands(signal,fs)
            if @_ all(0 ≈ _,signal)
                result[:,Between(:delta,:gamma)] .= 0
            end
            result
        end
    @save cachefile freqmeans
    alert()
else
    @load cachefile freqmeans
end

powerdf = @_ freqmeans |>
    filter(_.condition in [:global,:object],__) |>
    filter(_.hit == :hit,__) |>
    stack(__, Between(:delta,:gamma),
        variable_name = :freqbin, value_name = :power) |>
    filter(all(!isnan,_.power), __)

ε = max(1e-8,minimum(filter(!iszero,powerdf.power))/2)
powerdiff_df = @_ powerdf |>
    unstack(__, :window_timing, :power) |>
    by(__, [:sid,:hit,:freqbin,:condition,:winstart,:winlen,:channel,:salience],
        (:before,:after) => sdf ->
            (powerdiff = mean(log.(ε .+ sdf.after) .- log.(ε .+ sdf.before)),))

powerdiff_df[!,:channel_bin] .=
    categorical(Symbol.(:channel_,powerdiff_df.channel,:_,powerdiff_df.freqbin))

classdf = @_ powerdiff_df |>
    unstack(__, [:sid, :condition, :winstart, :winlen, :salience],
        :channel_bin, :powerdiff) |>
    filter(all(!ismissing,_[r"channel"]), __) |>
    disallowmissing!(__,r"channel")


# TODO: for some reason we still have infinite values in powerdiff
# figure that out and then create the plot
classpredict = by(classdf, [:winstart,:winlen,:salience]) do sdf
    labels = testmodel(NuSVC(),sdf,:sid,:condition,r"channel")
    DataFrame(correct = sdf.condition .== labels,sid = sdf.sid)
end

subj_means = @_ classpredict |>
    by(__,[:winstart,:winlen,:salience],:correct => mean)

sort!(subj_means,order(:correct_mean,rev=true))
first(subj_means,6)

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

subj_means.llen = log.(2,subj_means.winlen)
subj_means.lstart = log.(2,subj_means.winstart)

pl = subj_means |>
    @vlplot(:rect,
        x={
            field=:lstart,
            bin={step=4/9,anchor=-3-2/9},
        },
        y={
            field=:llen,
            bin={step=4/9,anchor=-3-2/9},
        },
        color={:correct_mean,scale={reverse=true,domain=[0.5,1],scheme="plasma"}},
        column=:salience,
        row={field=:freqbin,type=:ordinal,
             sort=[:delta,:theta,:alpha,:beta,:gamma]})

save(File(format"PDF",joinpath(dir,"hitonly_svm_allbins.pdf")),pl)


freqbins = OrderedDict(
    :delta => (1,3),
    :theta => (3,7),
    :alpha => (7,15),
    :beta => (15,30),
    :gamma => (30,100),
)

powerdiff_df.hit = string.(powerdiff_df.hit)
powerdiff_df.freqbin = string.(powerdiff_df.freqbin)
powerdiff_df.condition = string.(powerdiff_df.condition)
powerdiff_df.salience = string.(powerdiff_df.salience)
powerdiff_df.hit_channel_bin = string.(powerdiff_df.hit_channel_bin)

# look at relevant windows
R"""
library(tidyr)
library(dplyr)
library(ggplot2)

bins = $(collect(keys(freqbins)))

df = $powerdiff_df %>% filter(winlen == 2,winstart == 2) %>%
    mutate(freqbin = factor(freqbin,levels=bins, ordered=T)) %>%
    arrange(freqbin) %>%
    group_by(hit,freqbin,condition,salience,sid,channel) %>%
    summarize(powerdiff = mean(powerdiff))

plotdf = df %>% filter(hit != 'baseline', channel == 1)

ggplot(plotdf,aes(x=freqbin,y=powerdiff,color=hit,group=hit)) +
    facet_grid(condition~salience) +
    stat_summary(fun.data='mean_cl_boot',geom='pointrange',size=0.5,fun.args=list(conf.int=0.75)) +
    scale_color_brewer(palette='Set1') +
    geom_abline(intercept=0,slope=0,linetype=2)

cond_df = df %>% group_by(hit,freqbin,salience,sid,channel) %>%
    spread(condition, powerdiff) %>%
    mutate(cond_diff = global - object) %>%
    filter(hit != 'baseline', channel <= 5)

hit_df = cond_df %>%
    group_by(freqbin,salience,sid,channel) %>%
    select(-global,-object) %>%
    spread(hit, cond_diff) %>%
    mutate(hit_diff = hit - miss)

pos = position_dodge(width=0.2)
ggplot(hit_df,aes(x=freqbin,y=hit_diff,color=salience,group=salience)) +
    facet_grid(channel~.) +
    stat_summary(fun.data='mean_cl_boot',geom='pointrange',size=0.5,
        position=pos,
        fun.args=list(conf.int=0.75)) +
    scale_color_brewer(palette='Set1') +
    geom_abline(intercept=0,slope=0,linetype=2)

ggsave(file.path($dir,"hit_difference_all_bins.pdf"))


cond_df2 = df %>%
    filter(hit != 'baseline', channel <= 5) %>%
    group_by(condition,freqbin,salience,sid,channel) %>%
    spread(hit, powerdiff) %>%
    mutate(hit_diff = hit - miss)


hit_df2 = cond_df2 %>%
    group_by(freqbin,salience,sid,channel) %>%
    select(-hit,-miss) %>%
    spread(condition, hit_diff) %>%
    mutate(cond_diff = global - object)


pos = position_dodge(width=0.2)
ggplot(hit_df2,aes(x=freqbin,y=cond_diff,color=salience,group=salience)) +
    facet_grid(channel~.) +
    stat_summary(fun.data='mean_cl_boot',geom='pointrange',size=0.5,
        position=pos,
        fun.args=list(conf.int=0.75)) +
    scale_color_brewer(palette='Set1') +
    geom_abline(intercept=0,slope=0,linetype=2)

ggsave(file.path($dir,"hit_difference_v2_all_bins.pdf")) # basically identical

plotdf = read.csv("temp.csv") %>%
    filter(abs(winlen - 0.42) < 0.02,abs(winstart - 0.125) < 0.02) %>%
    mutate(freqbin = factor(freqbin,levels=bins, ordered=T))

plotdf = plotdf %>% filter(hit != 'baseline')

ggplot(plotdf,aes(x=freqbin,y=powerdiff,color=hit,group=hit)) +
    facet_grid(condition~salience) +
    stat_summary(fun.data='mean_cl_boot',geom='pointrange',size=0.5,fun.args=list(conf.int=0.75)) +
    scale_color_brewer(palette='Set1') +
    geom_abline(intercept=0,slope=0,linetype=2)

ggsave(file.path($dir,"powerdiff_len0.42_start0.125.pdf"))
"""

highvlow = @_ subj_means |>
    unstack(__,:salience,:correct_mean) |>
    by(__, [:winstart,:winlen],
        (:low,:high) => row -> (diff = row.low - row.high,))

highvlow.llen = log.(2,highvlow.winlen)
highvlow.lstart = log.(2,highvlow.winstart)

pl = highvlow |>
    @vlplot(:rect,
        x={
            field=:lstart,
            bin={step=4/9,anchor=-3-2/9},
        },
        y={
            field=:llen,
            bin={step=4/9,anchor=-3-2/9},
        },
        color={field=:diff,scale={domain=[-0.5,0.5], scheme="redblue"}},
        row={field=:freqbin,type=:ordinal,
             sort=[:delta,:theta,:alpha,:beta,:gamma]})

save(File(format"PDF",joinpath(dir,"freqbin_feat_diff_svm_allbins.pdf")),pl)

powerdf = @_ freqmeans |>
    filter(_.condition in [:global,:spatial],__) |>
    stack(__, Between(:delta,:gamma),
        variable_name = :freqbin, value_name = :power) |>
    filter(all(!isnan,_.power), __)

ε = max(1e-8,minimum(filter(!iszero,powerdf.power))/2)
powerdiff_df = @_ powerdf |>
    unstack(__, :window_timing, :power) |>
    by(__, [:sid,:hit,:freqbin,:condition,:winstart,:winlen,:channel,:salience],
        (:before,:after) => sdf ->
            (powerdiff = mean(log.(ε .+ sdf.after) .- log.(ε .+ sdf.before)),))

powerdiff_df[!,:hit_channel] .=
    categorical(Symbol.(:channel_,powerdiff_df.channel,:_,powerdiff_df.hit))
classdf = @_ powerdiff_df |>
    unstack(__, [:sid, :freqbin, :condition, :winstart, :winlen, :salience],
        :hit_channel, :powerdiff) |>
    filter(all(!ismissing,_[r"channel"]), __) |>
    disallowmissing!(__,r"channel")


# TODO: for some reason we still have infinite values in powerdiff
# figure that out and then create the plot
classpredict = by(classdf, [:freqbin,:winstart,:winlen,:salience]) do sdf
    labels = testmodel(NuSVC(),sdf,:sid,:condition,r"channel")
    DataFrame(correct = sdf.condition .== labels,sid = sdf.sid)
end

subj_means = @_ classpredict |>
    by(__,[:winstart,:winlen,:freqbin,:salience],:correct => mean)

sort!(subj_means,order(:correct_mean,rev=true))
first(subj_means,6)

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

subj_means.llen = log.(2,subj_means.winlen)
subj_means.lstart = log.(2,subj_means.winstart)

pl = subj_means |>
    @vlplot(:rect,
        x={
            field=:lstart,
            bin={step=4/9,anchor=-3-2/9},
        },
        y={
            field=:llen,
            bin={step=4/9,anchor=-3-2/9},
        },
        color={:correct_mean,scale={reverse=true,domain=[0.5,1],scheme="plasma"}},
        column=:salience,
        row={field=:freqbin,type=:ordinal,
             sort=[:delta,:theta,:alpha,:beta,:gamma]})

save(File(format"PDF",joinpath(dir,"spatial_svm_allbins.pdf")),pl)

CSV.write("temp.csv",powerdiff_df)
# look at relevant windows
R"""
library(dplyr)
library(ggplot2)

bins = $(collect(keys(freqbins)))

plotdf = read.csv("temp.csv") %>% filter(winlen == 2,winstart == 2) %>%
    mutate(freqbin = factor(freqbin,levels=bins, ordered=T)) %>%
    arrange(freqbin)

plotdf = plotdf %>% filter(hit != 'baseline')

ggplot(plotdf,aes(x=freqbin,y=powerdiff,color=hit,group=hit)) +
    facet_grid(condition~salience) +
    stat_summary(fun.data='mean_cl_boot',geom='pointrange',size=0.5,fun.args=list(conf.int=0.75)) +
    scale_color_brewer(palette='Set1') +
    geom_abline(intercept=0,slope=0,linetype=2)

ggsave(file.path($dir,"spatial_powerdiff_len2_start2.pdf"))

plotdf = read.csv("temp.csv") %>%
    filter(abs(winlen - 0.42) < 0.02,abs(winstart - 0.125) < 0.02) %>%
    mutate(freqbin = factor(freqbin,levels=bins, ordered=T))

plotdf = plotdf %>% filter(hit != 'baseline')

ggplot(plotdf,aes(x=freqbin,y=powerdiff,color=hit,group=hit)) +
    facet_grid(condition~salience) +
    stat_summary(fun.data='mean_cl_boot',geom='pointrange',size=0.5,fun.args=list(conf.int=0.75)) +
    scale_color_brewer(palette='Set1') +
    geom_abline(intercept=0,slope=0,linetype=2)

ggsave(file.path($dir,"spatial_powerdiff_len0.42_start0.125.pdf"))
"""

highvlow = @_ subj_means |>
    unstack(__,:salience,:correct_mean) |>
    by(__, [:winstart,:winlen,:freqbin],
        (:low,:high) => row -> (diff = row.low - row.high,))

highvlow.llen = log.(2,highvlow.winlen)
highvlow.lstart = log.(2,highvlow.winstart)

pl = highvlow |>
    @vlplot(:rect,
        x={
            field=:lstart,
            bin={step=4/9,anchor=-3-2/9},
        },
        y={
            field=:llen,
            bin={step=4/9,anchor=-3-2/9},
        },
        color={field=:diff,scale={domain=[-0.5,0.5], scheme="redblue"}},
        row={field=:freqbin,type=:ordinal,
             sort=[:delta,:theta,:alpha,:beta,:gamma]})

save(File(format"PDF",joinpath(dir,"spatial_diff_svm_allbins.pdf")),pl)

