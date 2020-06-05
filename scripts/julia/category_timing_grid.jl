# ----------------------------------- Setup ---------------------------------- #

using DrWatson
@quickactivate("german_track")
use_cache = true
seed = 072189
use_slurm = gethostname() == "lcap.cluster"

using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures,
    Dates, Underscores, Random, Printf, ProgressMeter, VegaLite, FileIO,
    StatsBase, Bootstrap, BangBang, Transducers, PyCall, ScikitLearn, Flux

# local only packages
using Formatting

import GermanTrack: stim_info, speakers, directions, target_times, switch_times

# we parallelize model parameter optimization with multi-process
# computing; I'd love to use multi-threading, but the current model
# is a python implementaiton and PyCall doesn't support multi-threading
# it's not even clear that it is technically feasible given the python GIL
using Distributed
@static if use_slurm
    using ClusterManagers
    if !(nprocs() > 1)
        addprocs(SlurmManager(10), partition="CPU", t="16:00:00", mem="32G",
            exeflags="--project=.")
    end
else
    # if !(nprocs() > 1)
    #     addprocs(4,exeflags="--project=.")
    # end
end

@everywhere begin
    seed = 072189

    using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures,
        Dates, Underscores, StatsBase, Random, Printf, ProgressMeter, VegaLite,
        FileIO, StatsBase, Bootstrap, BangBang, Transducers, PyCall,
        PyCall, ScikitLearn, Flux

    import GermanTrack: stim_info, speakers, directions, target_times, switch_times
end

@everywhere( @sk_import svm: (NuSVC, SVC) )

if !use_slurm
    dir = joinpath(plotsdir(),string("results_",Date(now())))
    isdir(dir) || mkdir(dir)
end

# ------------------------ Mean Frequency Bin Analysis ----------------------- #

classdf_file = joinpath(cache_dir(),"data","freqmeans_target_time.csv")
if use_cache && isfile(classdf_file)
    classdf = CSV.read(classdf_file)
else
    eeg_files = dfhit = @_ readdir(data_dir()) |> filter(occursin(r".mcca$",_), __)
    subjects = Dict(file => load_subject(joinpath(data_dir(), file), stim_info,
                                        encoding = RawEncoding())
        for file in eeg_files)
    classdf = find_powerdiff(
        subjects,groups=[:target_time],
        hittypes = ["hit"],
        regions = ["target"],
        windows = [(len=len,start=start,before=-len)
            for len in 2.0 .^ range(-1,1,length=10),
                start in [0; 2.0 .^ range(-2,2,length=9)]])
    CSV.write(classdf_file,classdf)
end

# --------------- Hyper-parameter Optimization: Global v Object -------------- #

objectdf = @_ classdf |> filter(_.condition in ["global","object"],__)

@everywhere begin
    np = pyimport("numpy")
    _wmean(x,weight) = (sum(x.*weight) + 1) / (sum(weight) + 2)

    function modelacc((key,sdf),params)
        # some values of nu may be infeasible, so we have to
        # catch those and return the worst possible fitness
        try
            np.random.seed(typemax(UInt32) & hash((params,seed)))
            result = testmodel(sdf,NuSVC(;params...),
                :sid,:condition,r"channel",n_folds=3)
            μ = _wmean(result.correct,result.weight)
            return (mean = μ, NamedTuple(key)...)
        catch e
            if e isa PyCall.PyError
                @info "Error while evaluting function: $(e)"
                return (mean = 0, NamedTuple(key)...)
            else
                rethrow(e)
            end
        end
    end
end

param_range = (nu=(0.0,0.95),gamma=(-4.0,1.0))
param_by = (nu=identity,gamma=x -> 10^x)
opts = (
    by=param_by,
    MaxFuncEvals = 1_500,
    # MaxFuncEvals = 6,
    FitnessTolerance = 0.03,
    TargetFitness = 0.0,
    # PopulationSize = 25,
)

paramdir = joinpath(datadir(),"svm_params")
isdir(paramdir) || mkdir(paramdir)
paramfile = joinpath(paramdir,"object_target_time.csv")
n_folds = 5
if !use_cache || !isfile(paramfile)
    progress = Progress(opts.MaxFuncEvals*n_folds,"Optimizing object params...")
    let result = Empty(DataFrame)
        for (i,(train,test)) in enumerate(folds(n_folds,objectdf.sid |> unique))
            Random.seed!(hash((seed,:object,i)))
            fold_params, fitness = optparams(param_range;opts...) do params
                gr = @_ objectdf |> filter(_.sid ∈ train,__) |>
                    groupby(__, [:winstart,:winlen,:target_time]) |>
                    pairs |> collect

                subresult = dreduce(append!!,Map(i -> [modelacc(gr[i],params)]),
                    1:length(gr),init=Empty(Vector))

                next!(progress)

                if subresult isa Vector{<:NamedTuple}
                    maxacc = @_ DataFrame(subresult) |>
                        groupby(__,:target_time) |>
                        combine(:mean => maximum => :max,__)
                    return 1 - mean(maxacc.max)#/length(gr)
                else
                    @info "Exception: $subresult"
                    return 1.0
                end
            end
            fold_params = GermanTrack.apply(param_by,fold_params)
            result = append!!(result,DataFrame(subjects = test; fold_params...))
        end

        ProgressMeter.finish!(progress)
        CSV.write(paramfile, result)
        global best_params = result
    end
else
    best_params = @_ CSV.read(paramfile)
    rename!(best_params,:subjects => :sid)
end

# ----------------------- Object Classification Results ---------------------- #

if !use_slurm
    @everywhere function modelresult((key,sdf))
        params = (nu = key[:nu], gamma = key[:gamma])
        np.random.seed(typemax(UInt32) & hash((params,seed)))
        testmodel(sdf,NuSVC(;params...),:sid,:condition,r"channel")
    end

    testgroups = @_ objectdf |>
        innerjoin(__,best_params,on=:sid) |>
        groupby(__, [:winstart,:winlen,:target_time,:nu,:gamma])
    object_classpredict = dreduce(append!!,Map(modelresult),
        collect(pairs(testgroups)),init=Empty(DataFrame))

    subj_means = @_ object_classpredict |>
        groupby(__,[:winstart,:winlen,:target_time,:sid]) |>
        combine(__,[:correct,:weight] => _wmean => :correct)
    wimeans = @_ subj_means |>
        groupby(__,[:winstart,:winlen,:target_time]) |>
        combine(__,:correct => mean)

    sort!(wimeans,order(:correct_mean,rev=true))
    first(wimeans,6)

    dir = joinpath(plotsdir(),string("results_",Date(now())))
    isdir(dir) || mkdir(dir)

    wimeans.llen = log.(2,wimeans.winlen)
    wimeans.lstart = log.(2,wimeans.winstart)

    pl = wimeans |>
        @vlplot(:rect,
            x={ field=:lstart, bin={step=0.573}, },
            y={ field=:llen, bin={step=2/9}, },
            color={
                :correct_mean,
                scale={reverse=true,domain=[0.5,1],scheme="plasma"}
            },
            column=:target_time)

    save(joinpath(dir,"object_target_time.pdf"),pl)
end

# -------------- Hyper-parameter Optimization: Global v Spatial -------------- #

spatialdf = @_ classdf |> filter(_.condition in ["global","spatial"],__)

paramdir = joinpath(datadir(),"svm_params")
isdir(paramdir) || mkdir(paramdir)
paramfile = joinpath(paramdir,"spatial_target_time.csv")
n_folds = 5
if use_slurm || !use_cache || !isfile(paramfile)
    progress = Progress(opts.MaxFuncEvals*n_folds,"Optimizing spatial params...")
    let result = Empty(DataFrame)
        for (i,(train,test)) in enumerate(folds(n_folds,spatialdf.sid |> unique))
            Random.seed!(hash((seed,:spatial,i)))
            fold_params, fitness = optparams(param_range;opts...) do params
                gr = @_ spatialdf |> filter(_.sid ∈ train,__) |>
                    groupby(__, [:winstart,:winlen,:target_time]) |>
                    pairs |> collect

                subresult = dreduce(append!!,Map(i -> [modelacc(gr[i],params)]),
                    1:length(gr),init=Empty(Vector))

                next!(progress)

                if subresult isa Vector{<:NamedTuple}
                    maxacc = @_ DataFrame(subresult) |>
                        groupby(__,:target_time) |>
                        combine(:mean => maximum => :max,__)
                    return 1 - mean(maxacc.max)#/length(gr)
                else
                    @info "Exception: $subresult"
                    return 1.0
                end
            end
            fold_params = GermanTrack.apply(param_by,fold_params)
            result = append!!(result,DataFrame(subjects = test; fold_params...))
        end

        ProgressMeter.finish!(progress)
        CSV.write(paramfile, result)
        global best_params = result
    end
else
    best_params = CSV.read(paramfile)
    rename!(best_params,:subjects => :sid)
end

# ----------------------- Object Classification Results ---------------------- #

if !use_slurm
    @everywhere function modelresult((key,sdf))
        params = (nu = key[:nu], gamma = key[:gamma])
        np.random.seed(typemax(UInt32) & hash((params,seed)))
        testmodel(sdf,NuSVC(;params...),:sid,:condition,r"channel")
    end

    testgroups = @_ spatialdf |>
        innerjoin(__,best_params,on=:sid) |>
        groupby(__, [:winstart,:winlen,:target_time,:nu,:gamma])
    spatial_classpredict = dreduce(append!!,Map(modelresult),
        collect(pairs(testgroups)),init=Empty(DataFrame))

    subj_means = @_ spatial_classpredict |>
        groupby(__,[:winstart,:winlen,:target_time,:sid]) |>
        combine(__,[:correct,:weight] => _wmean => :correct)
    wimeans = @_ subj_means |>
        groupby(__,[:winstart,:winlen,:target_time]) |>
        combine(__,:correct => mean)

    sort!(wimeans,order(:correct_mean,rev=true))
    first(wimeans,6)

    dir = joinpath(plotsdir(),string("results_",Date(now())))
    isdir(dir) || mkdir(dir)

    wimeans.llen = log.(2,wimeans.winlen)
    wimeans.lstart = log.(2,wimeans.winstart)

    pl = wimeans |>
        @vlplot(:rect,
            x={ field=:lstart, bin={step=0.573}, },
            y={ field=:llen, bin={step=2/9}, },
            color={
                :correct_mean,
                scale={reverse=true,domain=[0.5,1],scheme="plasma"}
            },
            column=:target_time)

    save(joinpath(dir,"spatial_target_time.pdf"),pl)
end

# -------------------------- Find Best Window Length ------------------------- #

@static if !use_slurm

    object_winlen_means = @_ object_classpredict |>
        groupby(__,[:winstart,:winlen,:target_time,:sid]) |>
        combine(__,[:correct,:weight] => _wmean => :correct) |>
        groupby(__,[:winlen,:target_time]) |>
        combine(__,:correct => mean) |>
        insertcols!(__,:condition => "object")

    spatial_winlen_means = @_ spatial_classpredict |>
        groupby(__,[:winstart,:winlen,:target_time,:sid]) |>
        combine(__,[:correct,:weight] => _wmean => :correct) |>
        groupby(__,[:winlen,:target_time]) |>
        combine(__,:correct => mean) |>
        insertcols!(__,:condition => "spatial")

    best_windows = @_ vcat(object_winlen_means,spatial_winlen_means) |>
        groupby(__,[:target_time,:condition]) |>
        combine(__,[:winlen,:correct_mean] =>
            ((len,val) -> len[argmax(val)]) => :winlen)

    CSV.write(joinpath(datadir(),"svm_params","best_windows_target_time.csv"),best_windows)
end

# ------------------------------------ End ----------------------------------- #
