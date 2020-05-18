using DrWatson
@quickactivate("german_track")
use_cache = true

using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures,
    Dates, Underscores, StatsBase, Random, Printf, ProgressMeter, VegaLite,
    FileIO, StatsBase, RCall, Bootstrap, BangBang, Transducers, PyCall

# local only packages
using Formatting, ScikitLearn

import GermanTrack: stim_info, speakers, directions, target_times, switch_times

# we parallelize model parameter optimization with multi-process
# computing; I'd love to use multi-threading, but the current model
# is a python implementaiton and PyCall doesn't support multi-threading
# it's not even clear that'st technically feasible given the python GIL
using Distributed
addprocs(6,exeflags="--project=.")

@everywhere begin
    seed = 110983

    using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures,
        Dates, Underscores, StatsBase, Random, Printf, ProgressMeter, VegaLite,
        FileIO, StatsBase, RCall, Bootstrap, BangBang, Transducers, PyCall

    using ScikitLearn

    import GermanTrack: stim_info, speakers, directions, target_times, switch_times
end

@everywhere( @sk_import svm: (NuSVC, SVC) )

eeg_files = dfhit = @_ readdir(data_dir()) |> filter(occursin(r".mcca$",_), __)
subjects = Dict(file => load_subject(joinpath(data_dir(), file), stim_info,
                                    encoding = RawEncoding())
    for file in eeg_files)

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

# TODO: try running each window separatley and storing the
# results, rather than storing all versions of the data

best_windows = CSV.read(joinpath(datadir(),"svm_params","best_windows.csv"))

classdf_file = joinpath(cache_dir(),"data","freqmeans_timeline.csv")
if use_cache && isfile(classdf_file)
    classdf = CSV.read(classdf_file)
else
    classdf = find_powerdiff(
        subjects,groups=[:salience],
        hittypes = ["hit"],
        windows = [(len=len,start=start,before=-len)
            for start in range(0,8,length=80), len in best_windows.winlen |> unique])
    CSV.write(classdf_file,classdf)
end

winlens = groupby(best_windows,[:condition,:salience])
objectdf = @_ classdf |>
    filter(_.condition in ["global","object"],__) |>
    filter(_1.winlen == winlens[(condition = "object", salience = _1.salience)].winlen[1],__)

@everywhere begin
    using DrWatson
    np = pyimport("numpy")

    classdf_file = joinpath(cache_dir(),"data","freqmeans_timeline.csv")
    classdf = CSV.read(classdf_file)

    best_windows = CSV.read(joinpath(datadir(),"svm_params","best_windows.csv"))

    winlens = groupby(best_windows,[:condition,:salience])
    objectdf = @_ classdf |>
        filter(_.condition in ["global","object"],__) |>
        filter(_1.winlen == winlens[(condition = "object", salience = _1.salience)].winlen[1],__)
end

paramfile = joinpath(datadir(),"svm_params","object_salience.csv")
best_params = NamedTuple(first(CSV.read(paramfile)))

@everywhere function modelresult((key,sdf),params)
    np.random.seed(typemax(UInt32) & hash((params,seed)))
    result = testmodel(sdf,NuSVC(;params...),
        :sid,:condition,r"channel")
    foreach(kv -> result[!,kv[1]] .= kv[2],pairs(key))
    result
end
vset = @_ objectdf.sid |> unique |>
    StatsBase.sample(MersenneTwister(111820),__1,round(Int,0.2length(__1)))
testgroups = @_ objectdf |> filter(_.sid ∉ vset,__) |>
    groupby(__, [:winstart,:winlen,:salience])
object_classpredict = dreduce(append!!,Map(x -> modelresult(x,best_params)),
    collect(pairs(testgroups)),init=Empty(DataFrame))

subj_means = @_ object_classpredict |>
    groupby(__,[:winstart,:salience,:sid]) |>
    combine(__,:correct => mean)

# TODO: left off here - find error band and plot

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

band = @_ subj_means |>
    groupby(__,[:winstart,:salience]) |>
    combine(:correct_mean => function(correct)
        bs = bootstrap(mean,correct,BasicSampling(10_000))
        μ,low,high = 100 .* confint(bs,BasicConfInt(0.683))[1]
        (correct = μ, low = low, high = high)
    end,__)

pl = band |>
    @vlplot() +
    @vlplot(:line, x=:winstart, y=:correct, color=:salience) +
    @vlplot(:errorband, x=:winstart, y=:low, y2=:high, color=:salience)

save(joinpath(dir,"object_salience_timeline.pdf"),pl)

@everywhere begin
    best_windows = CSV.read(joinpath(datadir(),"svm_params","best_windows.csv"))
    winlens = groupby(best_windows,[:condition,:salience])
    spatialdf = @_ classdf |>
        filter(_.condition in ["global","spatial"],__) |>
        filter(_1.winlen == winlens[(condition = "spatial", salience = _1.salience)].winlen[1],__)
end

vset = @_ spatialdf.sid |> unique |>
    StatsBase.sample(MersenneTwister(111820),__1,round(Int,0.2length(__1)))
valgroups = @_ spatialdf |> filter(_.sid ∈ vset,__) |>
    groupby(__, [:winstart,:winlen,:salience])

paramfile = joinpath(datadir(),"svm_params","spatial_salience.csv")
best_params = NamedTuple(first(CSV.read(paramfile)))

@everywhere function modelresult((key,sdf),params)
    np.random.seed(typemax(UInt32) & hash((params,seed)))
    result = testmodel(sdf,NuSVC(;params...),
        :sid,:condition,r"channel")
    foreach(kv -> result[!,kv[1]] .= kv[2],pairs(key))
    result
end
testgroups = @_ spatialdf |> filter(_.sid ∉ vset,__) |>
    groupby(__, [:winstart,:winlen,:salience])
spatial_classpredict = dreduce(append!!,Map(x -> modelresult(x,best_params)),
    collect(pairs(testgroups)),init=Empty(DataFrame))

subj_means = @_ spatial_classpredict |>
    groupby(__,[:winstart,:salience,:sid]) |>
    combine(__,:correct => mean)


band = @_ subj_means |>
    groupby(__,[:winstart,:salience]) |>
    combine(:correct_mean => function(correct)
        bs = bootstrap(mean,correct,BasicSampling(10_000))
        μ,low,high = 100 .* confint(bs,BasicConfInt(0.683))[1]
        (correct = μ, low = low, high = high)
    end,__)

pl = band |>
    @vlplot() +
    @vlplot(:line, x=:winstart, y=:correct, color=:salience) +
    @vlplot(:errorband, x=:winstart, y=:low, y2=:high, color=:salience)

save(joinpath(dir,"spatial_salience_timeline.pdf"),pl)

