# ----------------------------------- Setup ---------------------------------- #

using DrWatson
@quickactivate("german_track")
use_cache = true
seed = 072189

using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures,
    Dates, Underscores, StatsBase, Random, Printf, ProgressMeter, VegaLite,
    FileIO, StatsBase, RCall, Bootstrap, BangBang, Transducers, PyCall

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

# TODO: try running each window separatley and storing the
# results, rather than storing all versions of the data

# ---------------------------- Freqmeans Analysis ---------------------------- #

best_windows = CSV.read(joinpath(datadir(),"svm_params","best_windows.csv"))

classdf_file = joinpath(cache_dir(),"data","freqmeans_timeline.csv")
if use_cache && isfile(classdf_file)
    classdf = CSV.read(classdf_file)
else
    eeg_files = dfhit = @_ readdir(data_dir()) |> filter(occursin(r".mcca$",_), __)
    subjects = Dict(file => load_subject(joinpath(data_dir(), file), stim_info,
                                            encoding = RawEncoding())
        for file in eeg_files)

    classdf = find_powerdiff(
        subjects,groups=[:salience],
        hittypes = ["hit"],
        regions = ["target"],
        windows = [(len=len,start=start,before=-len)
            for start in range(0,4,length=64),
                len in best_windows.winlen |> unique])

    CSV.write(classdf_file,classdf)
end

# ------------------------------ Timeline ----------------------------- #

function modelresult((key,sdf))
    params = (nu = key[:nu], gamma = key[:gamma])
    np.random.seed(typemax(UInt32) & hash((params,seed)))
    testmodel(sdf,NuSVC(;params...),:sid,:condition,r"channel")
end

_wmean(x,weight) = (sum(x.*weight) + 1) / (sum(weight) + 2)

function classpredict(df,params,condition)
    testgroups = @_ df |>
        innerjoin(__,params,on=:sid) |>
        groupby(__, [:winstart,:winlen,:salience,:nu,:gamma])
    predictions = foldl(append!!,Map(modelresult),
        collect(pairs(testgroups)),init=Empty(DataFrame))

    @_ predictions |>
        groupby(__,[:winstart,:salience,:sid]) |> #,:before]) |>
        combine(__,[:correct,:weight] => _wmean => :correct_mean) |>
        insertcols!(__,:condition => condition)
end

winlens = groupby(best_windows,[:condition,:salience])

objectdf = @_ classdf |>
    filter(_.condition in ["global","object"],__) |>
    filter(_1.winlen == winlens[(condition = "object", salience = _1.salience)].winlen[1],__)
paramfile = joinpath(datadir(),"svm_params","object_salience.csv")
best_params = CSV.read(paramfile)
rename!(best_params,:subjects => :sid)
object_predict = classpredict(objectdf, best_params, "object")

spatialdf = @_ classdf |>
    filter(_.condition in ["global","spatial"],__) |>
    filter(_1.winlen == winlens[(condition = "spatial", salience = _1.salience)].winlen[1],__)
paramfile = joinpath(datadir(),"svm_params","spatial_salience.csv")
best_params = CSV.read(paramfile)
rename!(best_params,:subjects => :sid)
spatial_predict = classpredict(spatialdf, best_params, "spatial")

predict = vcat(object_predict,spatial_predict)

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

band = @_ predict |>
    # filter(_.before == "zero",__) |>
    groupby(__,[:winstart,:salience,:condition]) |> #,:before]) |>
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
    geom_line() + facet_grid(~condition) +
    geom_abline(slope=0,intercept=50,linetype=2) +
    coord_cartesian(ylim=c(40,100))
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
    filter(_.winstart ∈ best_wins[(condition = _.condition,)].winstart, __) |>
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

# ----------------------------- Spatial Timeline ----------------------------- #

@everywhere begin
    best_windows = CSV.read(joinpath(datadir(),"svm_params","best_windows.csv"))
    winlens = groupby(best_windows,[:condition,:salience])
    spatialdf = @_ classdf |>
        filter(_.condition in ["global","spatial"],__) |>
        filter(_1.winlen == winlens[(condition = "spatial", salience = _1.salience)].winlen[1],__)
end

paramfile = joinpath(datadir(),"svm_params","spatial_salience.csv")
best_params = CSV.read(paramfile)

@everywhere function modelresult((key,sdf))
    params = (nu = key[:nu], gamma = key[:gamma])
    np.random.seed(typemax(UInt32) & hash((params,seed)))
    testmodel(sdf,NuSVC(;params...),:sid,:condition,r"channel")
end
rename!(best_params,:subjects => :sid)
testgroups = @_ spatialdf |>
    innerjoin(__,best_params,on=:sid) |>
    groupby(__, [:winstart,:winlen,:salience,:nu,:gamma])
spatial_classpredict = dreduce(append!!,Map(modelresult),
    collect(pairs(testgroups)),init=Empty(DataFrame))

subj_means = @_ spatial_classpredict |>
    groupby(__,[:winstart,:salience,:sid]) |>
    combine(__,[:correct,:weight] => _wmean => :correct_mean)


band = @_ subj_means |>
    groupby(__,[:winstart,:salience]) |>
    combine(:correct_mean => function(correct)
        bs = bootstrap(mean,correct,BasicSampling(10_000))
        μ,low,high = 100 .* confint(bs,BasicConfInt(0.683))[1]
        (correct = μ, low = low, high = high)
    end,__)

R"""

library(ggplot2)

pl = ggplot($band,aes(x=winstart,y=correct,color=salience)) +
    geom_ribbon(aes(ymin=low,ymax=high,fill=salience,color=NULL),alpha=0.4) +
    geom_line() +
    geom_abline(slope=0,intercept=50,linetype=2) +
    coord_cartesian(ylim=c(40,100))
pl

ggsave(file.path($dir,"spatial_salience_timeline.pdf"),pl,width=11,height=8)

"""

# ------------------------------------ End ----------------------------------- #
