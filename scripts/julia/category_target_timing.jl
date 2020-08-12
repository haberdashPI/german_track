# ----------------------------------- Setup ---------------------------------- #

using DrWatson
@quickactivate("greman_track")
use_cache = true
seed = 072189

using CSV, GermanTrack, EEGCoding, Underscores, DataFrames, Transducers,
    BangBang, ScikitLearn, RCall, Bootstrap, Statistics, Dates, PyCall

import GermanTrack: stim_info, speakers, directions, target_times, switch_times
@sk_import svm: (NuSVC, SVC)
np = pyimport("numpy")
R"library(ggplot2)"

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

# ----------------------------- Fremeans Analysis ---------------------------- #

best_windows = CSV.read(joinpath(datadir(),"classifier_params","best_windows.csv"))

classdf_file = joinpath(cache_dir(),"data","freqmeans_target_time.csv")
if use_cache && isfile(classdf_file)
    classdf = CSV.read(classdf_file)
else
    eeg_files = dfhit = @_ readdir(processed_datadir("eeg")) |> filter(occursin(r".mcca$",_), __)
    subjects = Dict(file => load_subject(joinpath(processed_datadir("eeg"), file), stim_info,
                                            encoding = RawEncoding())
        for file in eeg_files)

    classdf = find_powerdiff(
        subjects, groups=[:target_time],
        hittypes = ["miss","hit"],
        regions = ["target"],
        windows = [(len=len,start=start,before=-len)
            for start in range(0,4,length=64),
                len in best_windows.winlen |> unique])

    CSV.write(classdf_file,classdf)
end

# ------------------------------ Object Timeline ----------------------------- #

winlens = groupby(best_windows,[:condition,:salience])
objectdf = @_ classdf |>
    filter(_.condition in ["global","object"],__) |>
    filter(_1.winlen == winlens[(condition = "object", salience = _1.salience)].winlen[1],__)

paramfile = joinpath(datadir(),"classifier_params","object_salience.csv")
best_params = CSV.read(paramfile)


paramfile = joinpath(datadir(),"classifier_params","object_salience.csv")
best_params = CSV.read(paramfile)
rename!(best_params,:subjects => :sid)

function modelresult((key,sdf))
    params = (nu = key[:nu], gamma = key[:gamma])
    np.random.seed(typemax(UInt32) & stablehash(params,seed))
    testclassifier(sdf,NuSVC(;params...),:sid,:condition,r"channel")
end
testgroups = @_ objectdf |>
    innerjoin(__,best_params,on=:sid) |>
    groupby(__, [:winstart,:hit,:winlen,:salience,:nu,:gamma])
object_classpredict = foldl(append!!,Map(modelresult),
    collect(pairs(testgroups)),init=Empty(DataFrame))

_wmean(x,weight) = (sum(x.*weight) + 1) / (sum(weight) + 2)
subj_means = @_ object_classpredict |>
    groupby(__,[:winstart,:salience,:sid,:hit]) |> #,:before]) |>
    combine(__,[:correct,:weight] => _wmean => :correct_mean)

band = @_ subj_means |>
    # filter(_.before == "zero",__) |>
    groupby(__,[:winstart,:salience,:hit]) |> #,:before]) |>
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
    geom_line() +
    facet_grid(~hit) +
    geom_abline(slope=0,intercept=50,linetype=2) +
    coord_cartesian(ylim=c(40,100))
"""

R"""
ggsave(file.path($dir,"object_with_miss_salience_timeline.pdf"),pl,width=11,height=8)
"""

# ------------------------------------ End ----------------------------------- #
