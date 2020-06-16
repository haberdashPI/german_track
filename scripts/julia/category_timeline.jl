# ----------------------------------- Setup ---------------------------------- #

using DrWatson
@quickactivate("german_track")
use_cache = true
seed = 072189

using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures,
    Dates, Underscores, StatsBase, Random, Printf, ProgressMeter, VegaLite,
    FileIO, StatsBase, RCall, Bootstrap, BangBang, Transducers, PyCall,
    Distributions, Alert, JSON3, JSONTables

# local only packages
using Formatting, ScikitLearn, Distributions

import GermanTrack: stim_info, speakers, directions, target_times, switch_times

@sk_import svm: (NuSVC, SVC)

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

np = pyimport("numpy")
R"""
library(ggplot2)
library(cowplot)
library(dplyr)
"""

# ---------------------------- Freqmeans Analysis ---------------------------- #

best_windows = CSV.read(joinpath(data_dir(),"svm_params","best_windows.csv"))

spread(scale,npoints) = x -> spread(x,scale,npoints)
spread(x,scale,npoints) = quantile.(Normal(x,scale/2),range(0.05,0.95,length=npoints))

classdf_file = joinpath(cache_dir(),"data","freqmeans_timeline_sal_target_time_old.csv")
if use_cache && isfile(classdf_file)
    classdf = CSV.read(classdf_file)
else
    eeg_files = dfhit = @_ readdir(data_dir()) |> filter(occursin(r".mcca$",_), __)
    subjects = Dict(file => load_subject(joinpath(data_dir(), file), stim_info,
                                            encoding = RawEncoding())
        for file in eeg_files)

    classdf = find_powerdiff(
        subjects,groups=[:salience,:target_time],
        hittypes = ["hit"],
        regions = ["target"],
        windows = [(len=len,start=start,before=-len)
            for start in range(0,4,length=64),
                len in copy(MapCat(spread(0.5,6)),unique(best_windows.winlen))])

    CSV.write(classdf_file,classdf)
    alert("Salience Freqmeans Complete!")
end

paramdir = joinpath(data_dir(),"svm_params")
paramfile = joinpath(paramdir,savename("all-conds-salience-and-target",(;),"json"))
best_params = jsontable(open(JSON3.read,paramfile,"r")[:data]) |> DataFrame
if :subjects in propertynames(best_params) # some old files misnamed the sid column
    rename!(best_params,:subjects => :sid)
end

# --------------------------------- Timeline --------------------------------- #

function modelresult((key,sdf))
    params = (nu = key[:nu], gamma = key[:gamma])
    np.random.seed(typemax(UInt32) & hash((params,seed)))
    testmodel(sdf,NuSVC(;params...),:sid,:condition,r"channel")
end

function classpredict(df,params,condition,variables...)
    testgroups = @_ df |>
        innerjoin(__,params,on=:sid) |>
        groupby(__, [:winstart,:winlen,variables...,:nu,:gamma])
    predictions = foldl(append!!,Map(modelresult),
        collect(pairs(testgroups)),init=Empty(DataFrame))

    @_ predictions |>
        groupby(__,[:winstart,variables...,:sid]) |> #,:before]) |>
        combine(__,[:correct,:weight] => ((x,w) -> mean(x,weights(w.+1))) => :correct_mean) |>
        insertcols!(__,:condition => condition)
end

# TODO: get best_params from master branch
objectdf = @_ classdf |>
    filter(_.condition in ["global","object"],__)
object_predict = classpredict(objectdf, best_params, "object", :salience, :target_time)

spatialdf = @_ classdf |>
    filter(_.condition in ["global","spatial"],__)
spatial_predict = classpredict(spatialdf, best_params, "spatial", :salience, :target_time)

predict = vcat(object_predict,spatial_predict)

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

band = @_ predict |>
    # filter(_.before == "zero",__) |>
    groupby(__,[:winstart,:salience,:target_time,:condition]) |> #,:before]) |>
    combine(:correct_mean => function(correct)
        bs = bootstrap(mean,correct,BasicSampling(10_000))
        μ,low,high = 100 .* confint(bs,BasicConfInt(0.682))[1]
        (correct = μ, low = low, high = high)
    end,__) #|>
    # transform!(__,[:salience,:before] =>
    #     ((x,y) -> string.(x,"_",y)) => :salience_for)

R"""
pl = ggplot($band,aes(x=winstart,y=correct,color=salience)) +
    geom_ribbon(aes(ymin=low,ymax=high,fill=salience,color=NULL),alpha=0.4) +
    geom_line() + facet_grid(target_time~condition) +
    geom_abline(slope=0,intercept=50,linetype=2) +
    coord_cartesian(ylim=c(40,100))
pl
"""


R"""
ggsave(file.path($dir,"object_salience_timeline.pdf"),pl,width=11,height=8)
"""

best_wins = @_ predict |>
    groupby(__,[:winstart,:salience,:condition]) |>
    combine(__,:correct_mean => mean => :correct) |>
    groupby(__,[:salience,:condition]) |>
    combine(__,[:winstart,:correct] =>
        ((win,val) -> win[argmax(val)]) => :winstart) |>
    groupby(__,:condition)

subj_sum = @_ predict |>
    filter(_.winstart in best_wins[(condition = _.condition,)].winstart, __) |>
    groupby(__,[:winstart,:salience,:condition]) |>
    combine(:correct_mean => function(correct)
        bs = bootstrap(mean,correct,BasicSampling(10_000))
        μ,low,high = 100 .* confint(bs,BasicConfInt(0.682))[1]
        (correct = μ, low = low, high = high)
    end,__)

subj_sum = @_ subj_sum |>
    groupby(__,[:condition,:winstart]) |>
    transform!(__,[:condition,:winstart] => function(cond,start)
        best = best_wins[(condition = cond[1],)]
        bestfor = filter(x -> x.winstart == start[1],best)
        fill(bestfor.salience[1], length(start))
    end => :bestfor)

subj_sum = @_ subj_sum |>
    groupby(__,[:condition,:winstart,:bestfor]) |>
    transform!(__,[:winstart,:bestfor] => function(start,bestfor)
        @show start
        label = "Best Time \nfor $(uppercasefirst(bestfor[1]))\n"*
            "($(round(start[1],digits=2)) s)"
        fill(label,length(start))
    end => :label)

R"""
pl = plot_grid(
    ggplot(filter($subj_sum,condition == "object"),
        aes(x=label,y=correct,color=salience,group=salience)) +
        geom_bar(stat='identity',position=position_dodge(width=0.8),aes(fill=salience),width=0.6) +
        geom_linerange(aes(ymin=low,ymax=high),position=position_dodge(width=0.8),color='black') +
        guides(color=F,fill=F) + coord_cartesian(ylim=c(40,100)) +
        geom_abline(intercept = 50, slope = 0, linetype = 2) +
        xlab("") + ylab("% Correct"),
    ggplot(filter($subj_sum,condition == "spatial"),
        aes(x=label,y=correct,color=salience,group=salience)) +
        geom_bar(stat='identity',position=position_dodge(width=0.8),aes(fill=salience),width=0.6) +
        geom_linerange(aes(ymin=low,ymax=high),position=position_dodge(width=0.8),color='black') +
        coord_cartesian(ylim=c(40,100)) +
        geom_abline(intercept = 50, slope = 0, linetype = 2) +
        xlab("") + ylab("% Correct"),
        rel_widths = c(1,1.2),
    labels = c('Object', 'Spatial'), label_x=0.5, hjust=0.5)
"""

R"""
ggsave(file.path($dir,"salience_classify_summary.pdf"),pl,width=8,height=3)
"""

# ------------------------------ Overall vs Miss ----------------------------- #

classdf_file = joinpath(cache_dir(),"data","freqmeans_miss_baseline.csv")
if use_cache && isfile(classdf_file)
    classdf_missbase = CSV.read(classdf_file)
else
    eeg_files = dfhit = @_ readdir(data_dir()) |> filter(occursin(r".mcca$",_), __)
    subjects = Dict(file => load_subject(joinpath(data_dir(), file), stim_info,
                                            encoding = RawEncoding())
        for file in eeg_files)

    classdf_missbase = find_powerdiff(
        subjects,groups=[:salience],
        hittypes = ["miss","hit", "baseline"],
        regions = ["target", "baseline"],
        windows = [(len=len,start=start,before=-len)
            for start in range(0,4,length=64),
                len in best_windows_sal.winlen |> unique])

    CSV.write(classdf_file,classdf)
end

objectdf = @_ classdf_missbase |>
    filter(_.condition in ["global","object"],__) |>
    filter(_1.winlen == winlens[(condition = "object", salience = _1.salience)].winlen[1],__)
paramfile = joinpath(datadir(),"svm_params","object_salience.csv")
best_params = CSV.read(paramfile)
rename!(best_params,:subjects => :sid)
object_predict = classpredict(objectdf, best_params, "object", :salience, :hit)

spatialdf = @_ classdf_missbase |>
    filter(_.condition in ["global","spatial"],__) |>
    filter(_1.winlen == winlens[(condition = "spatial", salience = _1.salience)].winlen[1],__)
paramfile = joinpath(datadir(),"svm_params","spatial_salience.csv")
best_params = CSV.read(paramfile)
rename!(best_params,:subjects => :sid)
best_params = @_ best_params |>
    transform!(__,:nu => (nu -> clamp.(nu,0,0.7)) => :nu)
spatial_predict = classpredict(spatialdf, best_params, "spatial", :salience, :hit)

function bestonly(var,measure,df)
    means = combine(groupby(df,var),measure => mean => measure)
    bestvar = means[var][argmax(means[measure])]

    @_ filter(_[var] == bestvar,df)
end

hit_compare = @_ vcat(object_predict,spatial_predict) |>
    groupby(__,[:salience,:hit,:condition]) |>
    combine(bestonly(:winstart,:correct_mean,_),__) |>
    groupby(__,[:salience,:condition,:sid,:hit]) |>
    combine(__,:correct_mean => mean => :correct_mean) |>
    groupby(__,:hit) |>
    combine(:correct_mean => function(x)
        bs = bootstrap(mean,x,BasicSampling(10_000))
        μ,low,high = 100 .* confint(bs,BasicConfInt(0.682))[1]
        (correct = μ, low = low, high = high)
    end,__)

R"""
pl = ggplot($hit_compare,aes(x=factor(hit,order=T,levels=c('hit','miss','baseline')),
        y=correct,fill=hit)) +
    geom_bar(stat='identity',width=0.7) +
    geom_linerange(aes(ymin=low,ymax=high)) +
    coord_cartesian(ylim=c(40,100)) +
    geom_abline(intercept=50,slope=0,linetype=2) +
    guides(color=F,fill=F) +
    ylab('% Correct') + xlab('')
pl
"""

R"""
ggsave(file.path($dir,"hit_v_miss_v_baseline_beststart.pdf"),pl,width=11,height=8)
"""

hit_compare2 = @_ vcat(object_predict,spatial_predict) |>
    groupby(__,[:salience,:condition,:sid,:hit]) |>
    combine(__,:correct_mean => mean => :correct_mean) |>
    groupby(__,:hit) |>
    combine(:correct_mean => function(x)
        bs = bootstrap(mean,x,BasicSampling(10_000))
        μ,low,high = 100 .* confint(bs,BasicConfInt(0.682))[1]
        (correct = μ, low = low, high = high)
    end,__)

R"""
pl = ggplot($hit_compare2,aes(x=factor(hit,order=T,levels=c('hit','miss','baseline')),
        y=correct,fill=hit)) +
    geom_bar(stat='identity',width=0.7) +
    geom_linerange(aes(ymin=low,ymax=high)) +
    coord_cartesian(ylim=c(40,100)) +
    geom_abline(intercept=50,slope=0,linetype=2) +
    guides(color=F,fill=F) +
    ylab('% Correct') + xlab('')
pl
"""

R"""
ggsave(file.path($dir,"hit_v_miss_v_baseline_grouopavg.pdf"),pl,width=11,height=8)
"""

# -------------------------- Target Timing Timeline -------------------------- #

winlens = groupby(best_windows_tim,[:condition,:target_time])

objectdf = @_ classdf_tim |>
    filter(_.condition in ["global","object"],__) |>
    filter(_1.winlen in spread(winlens[(condition = "object", target_time = _1.target_time)].winlen[1],0.5,6),__)
paramfile = joinpath(datadir(),"svm_params","object_target_time.csv")
best_params = CSV.read(paramfile)
rename!(best_params,:subjects => :sid)
object_predict = classpredict(objectdf, best_params, "object", :target_time)

spatialdf = @_ classdf_tim |>
    filter(_.condition in ["global","spatial"],__) |>
    filter(_1.winlen in spread(winlens[(condition = "spatial", target_time = _1.target_time)].winlen[1],0.5,6),__)
paramfile = joinpath(datadir(),"svm_params","spatial_target_time.csv")
best_params = CSV.read(paramfile)
rename!(best_params,:subjects => :sid)
spatial_predict = classpredict(spatialdf, best_params, "spatial", :target_time)

predict = vcat(object_predict,spatial_predict)

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

band = @_ predict |>
    # filter(_.before == "zero",__) |>
    groupby(__,[:winstart,:target_time,:condition]) |> #,:before]) |>
    combine(:correct_mean => function(correct)
        bs = bootstrap(mean,correct,BasicSampling(10_000))
        μ,low,high = 100 .* confint(bs,BasicConfInt(0.682))[1]
        (correct = μ, low = low, high = high)
    end,__) #|>
    # transform!(__,[:target_time,:before] =>
    #     ((x,y) -> string.(x,"_",y)) => :target_time_for)

R"""
pl = ggplot($band,aes(x=winstart,y=correct,color=target_time)) +
    geom_ribbon(aes(ymin=low,ymax=high,fill=target_time,color=NULL),alpha=0.4) +
    geom_line() + facet_grid(~condition) +
    geom_abline(slope=0,intercept=50,linetype=2) +
    coord_cartesian(ylim=c(40,100))
"""

R"""
ggsave(file.path($dir,"object_target_time_timeline.pdf"),pl,width=11,height=8)
"""

best_wins = @_ predict |>
    groupby(__,[:winstart,:target_time,:condition]) |>
    combine(__,:correct_mean => mean => :correct) |>
    groupby(__,[:target_time,:condition]) |>
    combine(__,[:winstart,:correct] =>
        ((win,val) -> win[argmax(val)]) => :winstart) |>
    groupby(__,:condition)

subj_sum = @_ predict |>
    filter(_.winstart in best_wins[(condition = _.condition,)].winstart, __) |>
    groupby(__,[:winstart,:target_time,:condition]) |>
    combine(:correct_mean => function(correct)
        bs = bootstrap(mean,correct,BasicSampling(10_000))
        μ,low,high = 100 .* confint(bs,BasicConfInt(0.682))[1]
        (correct = μ, low = low, high = high)
    end,__)

subj_sum = @_ subj_sum |>
    groupby(__,[:condition,:winstart]) |>
    transform!(__,[:condition,:winstart] => function(cond,start)
        best = best_wins[(condition = cond[1],)]
        bestfor = filter(x -> x.winstart == start[1],best)
        fill(bestfor.target_time[1], length(start))
    end => :bestfor)

subj_sum = @_ subj_sum |>
    groupby(__,[:condition,:winstart,:bestfor]) |>
    transform!(__,[:winstart,:bestfor] => function(start,bestfor)
        @show start
        label = "Best Time \nfor $(uppercasefirst(bestfor[1]))\n"*
            "($(round(start[1],digits=2)) s)"
        fill(label,length(start))
    end => :label)

R"""
pl = plot_grid(
    ggplot(filter($subj_sum,condition == "object"),
        aes(x=label,y=correct,color=target_time,group=target_time)) +
        geom_bar(stat='identity',position=position_dodge(width=0.8),aes(fill=target_time),width=0.6) +
        geom_linerange(aes(ymin=low,ymax=high),position=position_dodge(width=0.8),color='black') +
        guides(color=F,fill=F) + coord_cartesian(ylim=c(40,100)) +
        geom_abline(intercept = 50, slope = 0, linetype = 2) +
        xlab("") + ylab("% Correct"),
    ggplot(filter($subj_sum,condition == "spatial"),
        aes(x=label,y=correct,color=target_time,group=target_time)) +
        geom_bar(stat='identity',position=position_dodge(width=0.8),aes(fill=target_time),width=0.6) +
        geom_linerange(aes(ymin=low,ymax=high),position=position_dodge(width=0.8),color='black') +
        coord_cartesian(ylim=c(40,100)) +
        geom_abline(intercept = 50, slope = 0, linetype = 2) +
        xlab("") + ylab("% Correct"),
        rel_widths = c(1,1.2),
    labels = c('Object', 'Spatial'), label_x=0.5, hjust=0.5)
"""

R"""
ggsave(file.path($dir,"target_time_classify_summary.pdf"),pl,width=8,height=3)
"""

# ------------------------------------ End ----------------------------------- #
