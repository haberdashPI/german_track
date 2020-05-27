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

# we parallelize model parameter optimization with multi-process
# computing; I'd love to use multi-threading, but the current model
# is a python implementaiton and PyCall doesn't support multi-threading
# it's not even clear that'st technically feasible given the python GIL
using Distributed
# addprocs(6,exeflags="--project=.")

@everywhere begin
    seed = 072189

    using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures,
        Dates, Underscores, StatsBase, Random, Printf, ProgressMeter, VegaLite,
        FileIO, StatsBase, RCall, Bootstrap, BangBang, Transducers, PyCall

    using ScikitLearn

    import GermanTrack: stim_info, speakers, directions, target_times, switch_times
end

@everywhere( @sk_import svm: (NuSVC, SVC) )

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

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
            for start in range(0,4,length=256),
                len in best_windows.winlen |> unique])

    CSV.write(classdf_file,classdf)
end

# classdf_rand_file = joinpath(cache_dir(),"data","freqmeans_timeline_randbefore.csv")
# if use_cache && isfile(classdf_rand_file)
#     classdf_rand = CSV.read(classdf_rand_file)
# else
#     Random.seed!(hash((:freqmeans,seed)))
#     classdf_rand = find_powerdiff(
#         subjects,groups=[:salience],
#         hittypes = ["hit"],
#         regions = ["target"],
#         windows = [(len = len, start = start,
#                     before = t -> t > len ? rand(Uniform(-t,-len-(t-len)/2)) : -len)
#             for start in range(0,4,length=160),
#                 len in best_windows.winlen |> unique])
#     CSV.write(classdf_rand_file,classdf_rand)
# end

# classdf_all = @_ insertcols!(classdf,:before => "zero") |>
#     vcat(__,insertcols!(classdf_rand,:before => "random"))

# ------------------------------ Object Timeline ----------------------------- #

winlens = groupby(best_windows,[:condition,:salience])
objectdf = @_ classdf |>
    filter(_.condition in ["global","object"],__) |>
    filter(_1.winlen == winlens[(condition = "object", salience = _1.salience)].winlen[1],__)

@everywhere begin
    using DrWatson
    np = pyimport("numpy")

    classdf_file = joinpath(cache_dir(),"data","freqmeans_timeline.csv")
    classdf = CSV.read(classdf_file)

    # classdf_rand_file = joinpath(cache_dir(),"data","freqmeans_timeline_randbefore.csv")
    # classdf_rand = CSV.read(classdf_rand_file)

    # classdf_all = @_ insertcols!(classdf,:before => "zero") |>
    #     vcat(__,insertcols!(classdf_rand,:before => "random"))

    best_windows = CSV.read(joinpath(datadir(),"svm_params","best_windows.csv"))

    winlens = groupby(best_windows,[:condition,:salience])
    objectdf = @_ classdf |>
        filter(_.condition in ["global","object"],__) |>
        filter(_1.winlen == winlens[(condition = "object", salience = _1.salience)].winlen[1],__)
end

paramfile = joinpath(datadir(),"svm_params","object_salience.csv")
best_params = CSV.read(paramfile)

@everywhere function modelresult((key,sdf))
    params = (nu = key[:nu], gamma = key[:gamma])
    np.random.seed(typemax(UInt32) & hash((params,seed)))
    testmodel(sdf,NuSVC(;params...),:sid,:condition,r"channel")
end
rename!(best_params,:subjects => :sid)
testgroups = @_ objectdf |>
    innerjoin(__,best_params,on=:sid) |>
    groupby(__, [:winstart,:winlen,:salience,:nu,:gamma])
object_classpredict = dreduce(append!!,Map(modelresult),
    collect(pairs(testgroups)),init=Empty(DataFrame))

_wmean(x,weight) = (sum(x.*weight) + 1) / (sum(weight) + 2)
subj_means = @_ object_classpredict |>
    groupby(__,[:winstart,:salience,:sid]) |> #,:before]) |>
    combine(__,[:correct,:weight] => _wmean => :correct_mean)

subj_counts = @_ object_classpredict |>
    groupby(__,[:winstart,:salience,:sid]) |> #,:before]) |>
    combine(__,:correct => length)
# TODO: left off here - find error band and plot

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

band = @_ subj_means |>
    # filter(_.before == "zero",__) |>
    groupby(__,[:winstart,:salience]) |> #,:before]) |>
    combine(:correct_mean => function(correct)
        bs = bootstrap(mean,correct,BasicSampling(10_000))
        μ,low,high = 100 .* confint(bs,BasicConfInt(0.682))[1]
        (correct = μ, low = low, high = high)
    end,__) #|>
    # transform!(__,[:salience,:before] =>
    #     ((x,y) -> string.(x,"_",y)) => :salience_for)

R"""

library(ggplot2)

pl = ggplot($band,aes(x=winstart,y=correct,color=salience)) +
    geom_ribbon(aes(ymin=low,ymax=high,fill=salience,color=NULL),alpha=0.4) +
    geom_line() +
    geom_abline(slope=0,intercept=50,linetype=2) +
    coord_cartesian(ylim=c(40,100))

ggsave(file.path($dir,"object_salience_timeline.pdf"),pl,width=11,height=8)

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
