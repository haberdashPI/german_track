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

cachefile = joinpath(cache_dir(),"..","data_cache","freqmeans.bson")
if !isfile(cachefile)
    freqmeans = organize_freqbands(subjects,groups=[:salience],hittypes = [:hit,:miss,:baseline],
        winlens = 2.0 .^ range(-3,1,length=10),
        winstarts = 2.0 .^ range(-3,1,length=10))
    @save cachefile freqmeans
    alert()
else
    @load cachefile freqmeans
end

powerdf = @_ freqmeans |>
    filter(_.condition in [:global,:object],__) |>
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

save(File(format"PDF",joinpath(dir,"svm_allbins.pdf")),pl)


freqbins = OrderedDict(
    :delta => (1,3),
    :theta => (3,7),
    :alpha => (7,15),
    :beta => (15,30),
    :gamma => (30,100),
)

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

ggsave(file.path($dir,"powerdiff_len2_start2.pdf"))

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

save(File(format"PDF",joinpath(dir,"diff_svm_allbins.pdf")),pl)

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

# TODO: why is high classification worse, early on
# try to reproduce old plot


powerdiff = @_ freqmeans |>
    filter((isapprox(_1.winstart,0.23,atol=0.02) &&
            isapprox(_1.winlen,0.58,atol=0.02)) ||
           (isapprox(_1.winstart,0.58,atol=0.02) &&
            isapprox(_1.winlen,1.46,atol=0.02)),__) |>
    filter(_.condition in [:global,:object],__) |>
    stack(__, [:delta,:theta,:alpha,:beta,:gamma],
        variable_name = :freqbin, value_name = :power) |>
    unstack(__, :window_timing, :power) |>
    by(__, [:sid,:hit,:freqbin,:condition,:winstart,:winlen,:channel,:salience],
        (:before,:after) => sdf ->
            (powerdiff = mean(log.(sdf.after) .- log.(sdf.before)),))

# powerdiff[!,:hit_channel] .=
#     categorical(Symbol.(map(x -> @sprintf("channel%02d",x),powerdiff.channel),
#         :_,powerdiff.hit))

powerdiff[!,:hit_channel] .=
    categorical(Symbol.(map(x -> @sprintf("channel%02d",x),powerdiff.channel)))

classdf = @_ powerdiff |>
    unstack(__, [:sid, :freqbin, :condition, :winstart, :winlen, :salience],
        :hit_channel, :powerdiff) |>
    filter(all(!ismissing,_[r"channel"]), __) |>
    disallowmissing!(__,r"channel")

classpredict = by(classdf, [:freqbin,:winstart,:winlen,:salience]) do sdf
    mapreduce(vcat,1:30) do channel
        labels = testmodel(LIBSVM.SVC(),sdf,:sid,:condition,Regex(@sprintf("channel%02d",channel)))
        DataFrame(correct = sdf.condition .== labels,sid = sdf.sid,channel = channel)
    end
end

classpredict[!,:channelgroup] = @_ map(@sprintf("channel%02d",_),classpredict.channel)

rnd = MersenneTwister(1983)
rseqs = [sort!(sample(rnd,1:30,5,replace=false)) for _ in 1:10]
channel_groups = OrderedDict(
    "1-5" => 1:5,
    "1-10" => 1:10,
    "1-20" => 1:20,
    "all" => 1:30,
    (join(r,",") => r for r in rseqs)...
)

for group in keys(channel_groups)
    chs = channel_groups[group]
    newrows = by(classdf,[:freqbin,:winstart,:winlen,:salience]) do sdf
        cols = Regex("("*join((@sprintf("channel%02d",ch) for ch in chs),"|")*")")
        labels = testmodel(LIBSVM.SVC(),sdf,:sid,:condition,cols)
        DataFrame(correct = sdf.condition .== labels,sid = sdf.sid)
    end
    newrows[!,:channelgroup] .= group
    newrows[!,:channel] .= maximum(classpredict.channel)+1
    append!(classpredict,newrows)
end
classpredict.correct = Int.(classpredict.correct)

classpredict |>
    @vlplot(:rect,
        x={:channel,type=:quantitative,bin={step=1}},
        y={:freqbin,type=:ordinal,
           sort=reverse([:delta,:theta,:alpha,:beta,:gamma])},
        column=:salience,
        color={"mean(correct)",type=:quantitative,scale={reverse=true,domain=[0.5,1],scheme="plasma"}},
        row=:winlen) #, column=:salience,row=:winlen)

mean_correct = by(classpredict,[:freqbin,:salience,:winlen,:channel],:correct => mean)
mean_correct.salience = string.(mean_correct.salience)
mean_correct.freqbin = string.(mean_correct.freqbin)
R"""

ggplot($mean_correct,aes(x=channel,y=freqbin,fill=correct_mean)) +
    geom_raster() + facet_grid(winlen~salience)
# TODO: try SVC, rather than NuSVC, try LIBSVM
# then try collapsing across hits, misses and baselines
"""
