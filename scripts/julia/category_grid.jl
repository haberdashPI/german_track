# Setup
# =================================================================

using DrWatson
@quickactivate("german_track")
use_cache = true
seed = 072189
num_local_procs = 1
num_cluster_procs = 16
use_absolute_features = true
use_slurm = gethostname() == "lcap.cluster"

using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures,
    Dates, Underscores, Random, Printf, ProgressMeter, VegaLite, FileIO,
    StatsBase, Bootstrap, BangBang, Transducers, PyCall, ScikitLearn, Flux,
    JSON3, JSONTables, Tables, Infiltrator, FileIO

DrWatson._wsave(file,data::Dict) = open(io -> JSON3.write(io,data), file, "w")

# local only pac kages
using Formatting

import GermanTrack: stim_info, speakers, directions, target_times, switch_times

# we parallelize model parameter optimization with multi-process
# computing; I'd love to use multi-threading, but the current model
# is a python implementaiton and PyCall doesn't support multi-threading
# it's not even clear that it is technically feasible given the python GIL
using Distributed
@static if use_slurm
    using ClusterManagers
    if !(nprocs() > 1) && num_cluster_procs > 1
        addprocs(SlurmManager(num_cluster_procs), partition="CPU", t="24:00:00", mem="32G",
            exeflags="--project=.")
    end
else
    if !(nprocs() > 1) && num_local_procs > 1
        addprocs(num_local_procs,exeflags="--project=.")
    end
end

@everywhere begin
    seed = 072189

    using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures,
    Dates, Underscores, Random, Printf, ProgressMeter, VegaLite, FileIO,
    StatsBase, Bootstrap, BangBang, Transducers, PyCall, ScikitLearn, Flux,
    JSON3, JSONTables, Tables, FileIO

    import GermanTrack: stim_info, speakers, directions, target_times, switch_times
end

@everywhere( @sk_import svm: (NuSVC, SVC) )

if !use_slurm
    dir = joinpath(plotsdir(),string("results_",Date(now())))
    isdir(dir) || mkdir(dir)
end

wmeanish(x,w) = iszero(sum(w)) ? 0.0 : mean(x,weights(w))

# is freq means always the same?

# Mean Frequency Bin Analysis
# =================================================================

if use_absolute_features
    classdf_file = joinpath(cache_dir(),"data","freqmeans_sal_and_target_time_absolute.csv")
else
    classdf_file = joinpath(cache_dir(),"data","freqmeans_sal_and_target_time.csv")
end

if use_cache && isfile(classdf_file)
    classdf = CSV.read(classdf_file)
else
    windows = [(len=len,start=start,before=-len)
        for len in 2.0 .^ range(-1,1,length=10),
            start in [0; 2.0 .^ range(-2,2,length=9)]]
    eeg_files = dfhit = @_ readdir(processed_datadir("eeg")) |> filter(occursin(r".mcca$",_), __)
    subjects = Dict(
        sidfor(file) => load_subject(
            joinpath(processed_datadir("eeg"), file), stim_info,
            encoding = RawEncoding()
        ) for file in eeg_files)

    events = @_ mapreduce(_.events,append!!,values(subjects))
    classdf_groups = @_ events |>
        filter(_.target_present,__) |>
        filter(ishit(_,region = "target") == "hit",__) |>
        groupby(__,[:salience_label,:target_time_label,:sid,:condition])

    progress = Progress(length(classdf_groups),desc="Computing frequency bins...")
    classdf = @_ classdf_groups |>
        combine(function(sdf)
            # compute features in each window
            x = mapreduce(append!!,windows) do window
                result = if use_absolute_features
                    compute_powerbin_features(subjects[sdf.sid[1]].eeg,sdf,"target",window)
                else
                    compute_powerdiff_features(subjects[sdf.sid[1]].eeg,sdf,"target",window)
                end
                result[!,:winstart] .= window.start
                result[!,:winlen] .= window.len
                result
            end
            next!(progress)
            x
        end,__)
    ProgressMeter.finish!(progress)
    CSV.write(classdf_file,classdf)
end

# Hyper-parameter Optimization: Global v Object
# =================================================================

objectdf = @_ classdf |> filter(_.condition in ["global","object"],__)
spatialdf = @_ classdf |> filter(_.condition in ["global","spatial"],__)

# Function Definitions
# -----------------------------------------------------------------

@everywhere begin
    np = pyimport("numpy")
    # _wmean(x,weight) = (sum(x.*weight) + 1) / (sum(weight) + 2)

    function resultmax(result,conditions...)
        if result isa Vector{<:NamedTuple}
            maxacc = @_ DataFrame(result) |>
                groupby(__,collect(conditions)) |>
                combine(__,:mean => maximum => :max)
            return mean(maxacc.max)#/length(gr)
        else
            @info "Exception: $result"
            return 0.0
        end
    end

    function modelacc((key,sdf),params)
        # some values of nu may be infeasible, so we have to
        # catch those and return the worst possible fitness
        try
            result = testclassifier(NuSVC(;params...), data = sdf,
                y = :condition,X = r"channel", crossval = :sid, n_folds=3,
                seed=hash((params,seed)))
            return (mean = wmeanish(result.correct,result.weight),
                weight = sum(result.weight),
                NamedTuple(key)...)
        catch e
            if e isa PyCall.PyError
                @info "Error while evaluting function: $(e)"
                return (mean = 0, weight = 0, NamedTuple(key)...)
            else
                rethrow(e)
            end
        end
    end
end

# Optimization
# -----------------------------------------------------------------

param_range = (nu=(0.0,0.5),gamma=(-4.0,1.0))
param_by = (nu=identity,gamma=x -> 10^x)
opts = (
    MaxFuncEvals = 1_500,
    # MaxFuncEvals = 6,
    FitnessTolerance = 0.03,
    TargetFitness = 0.0,
    # PopulationSize = 25,
)

# type piracy: awaiting PR acceptance to remove
JSON3.StructTypes.StructType(::Type{<:CategoricalValue{<:String}}) = JSON3.StructTypes.StringType()

paramdir = processed_datadir("svm_params")
isdir(paramdir) || mkdir(paramdir)
paramfile = joinpath(paramdir,savename("all-conds-salience-and-target",
    (absolute=use_absolute_features,),"json"))
n_folds = 5
if use_slurm || !use_cache || !isfile(paramfile)
    progress = Progress(opts.MaxFuncEvals*n_folds,"Optimizing params...")
    let result = Empty(DataFrame)
        for (i,(train,test)) in enumerate(folds(n_folds,objectdf.sid |> unique))
            Random.seed!(hash((seed,:object,i)))
            fold_params, fitness = optparams(param_range;opts...) do params

                tparams_vals = @_ map(_1(_2), param_by, params)
                tparams = NamedTuple{keys(param_by)}(tparams_vals)

                objectgr = @_ objectdf |> filter(_.sid ∈ train,__) |>
                    groupby(__, [:winstart,:winlen,:salience_label,:target_time_label]) |>
                    pairs |> collect

                objectresult = dreduce(append!!,
                    Map(i -> [modelacc(objectgr[i],tparams)]),
                    1:length(objectgr),init=Empty(Vector))

                spatialgr = @_ spatialdf |> filter(_.sid ∈ train,__) |>
                    groupby(__, [:winstart,:winlen,:salience_label,:target_time_label]) |>
                    pairs |> collect

                spatialresult = dreduce(append!!,
                    Map(i -> [modelacc(spatialgr[i],tparams)]),
                    1:length(spatialgr),init=Empty(Vector))

                # spatialresult = dreduce(append!!,
                #     Map(i -> [modelacc(spatialgr[i],params)]),
                #     1:length(spatialgr),init=Empty(Vector))

                next!(progress)

                maxacc = max(
                    resultmax(objectresult,:salience_label,:target_time_label),
                    resultmax(spatialresult,:salience_label,:target_time_label)
                )

                return 1.0 - maxacc
            end
            fold_params_vals = @_ map(_1(_2), param_by, fold_params)
            fold_params = NamedTuple{keys(param_by)}(fold_params_vals)
            result = append!!(result,DataFrame(sid = test; fold_params...))
        end

        ProgressMeter.finish!(progress)
        global best_params = result

        # save a reproducible record of the results
        @tagsave paramfile Dict(
            :data => JSONTables.ObjectTable(Tables.columns(best_params)),
            :seed => seed,
            :param_range => param_range,
            :optimize_parameters => Dict(k => v for (k,v) in pairs(opts) if k != :by)
        ) safe=true
    end
else
    global best_params = jsontable(open(JSON3.read,paramfile,"r")[:data]) |> DataFrame
    if :subjects in propertynames(best_params) # some old files misnamed the sid column
        rename!(best_params,:subjects => :sid)
    end
end

# Object Classification Results
# =================================================================

if !use_slurm

    @everywhere function modelresult((key,sdf))
        params = (nu = key[:nu], gamma = key[:gamma])
        testclassifier(NuSVC(;params...), data = sdf, y = :condition, X = r"channel",
            crossval = :sid, seed = hash((params, seed)))
    end

    testgroups = @_ objectdf |>
        innerjoin(__,best_params,on=:sid) |>
        groupby(__, [:winstart,:winlen,:salience_label,:target_time_label,:nu,:gamma])
    object_classpredict = dreduce(append!!,Map(modelresult),
        collect(pairs(testgroups)),init=Empty(DataFrame))

    subj_means = @_ object_classpredict |>
        groupby(__,[:winstart,:winlen,:salience_label,:target_time_label,:sid]) |>
        combine(__,[:correct,:weight] => wmeanish => :correct)
    wimeans = @_ subj_means |>
        groupby(__,[:winstart,:winlen,:salience_label,:target_time_label]) |>
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
            column=:salience_label,
            row=:target_time_label)

    if use_absolute_features
        save(joinpath(dir,"object_grid_absolute.pdf"),pl)
    else
        save(joinpath(dir,"object_grid.pdf"),pl)
    end
end

# Classifciation Results: Global v Spattial
# =================================================================

if !use_slurm

    @everywhere function modelresult((key,sdf))
        params = (nu = key[:nu], gamma = key[:gamma])
        if length(unique(sdf.condition)) == 1
            @info "Skipping data with one class: $(first(sdf,1))"
            Empty(DataFrame)
        else
            testclassifier(NuSVC(;params...), data = sdf, y = :condition, X = r"channel",
                crossval = :sid, seed = hash((params, seed)), n_folds = 3)
        end
    end

    testgroups = @_ spatialdf |>
        innerjoin(__,best_params,on=:sid) |>
        groupby(__, [:winstart,:winlen,:salience_label, :target_time_label,:nu,:gamma])
    spatial_classpredict = foldl(append!!,Map(modelresult),
        collect(pairs(testgroups)),init=Empty(DataFrame))

    subj_means = @_ spatial_classpredict |>
        groupby(__,[:winstart,:winlen,:salience_label, :target_time_label,:sid]) |>
        combine(__,[:correct,:weight] => wmeanish => :correct)
    wimeans = @_ subj_means |>
        groupby(__,[:winstart,:winlen,:salience_label,:target_time_label]) |>
        combine(__,:correct => mean)

    sort!(wimeans,order(:correct_mean,rev=true))
    first(wimeans,6)

    dir = joinpath(plotsdir(),string("results_",Date(now())))
    isdir(dir) || mkdir(dir)

    wimeans.llen = log.(2,wimeans.winlen)
    wimeans.lstart = log.(2,wimeans.winstart)

    pl = wimeans |>
        @vlplot(:rect,
            x={
                field=:lstart,
                bin={step=0.573},
            },
            y={
                field=:llen,
                bin={step=2/9},
            },
            color={:correct_mean,scale={reverse=true,domain=[0.5,1],scheme="plasma"}},
            column=:salience_label,row=:target_time_label)


    if use_absolute_features
        save(joinpath(dir,"spatial_grid_absolute.pdf"),pl)
    else
        save(joinpath(dir,"spatial_grid.pdf"),pl)
    end
end

# Find Best Window Length
# =================================================================

@static if !use_slurm

    object_winlen_means = @_ object_classpredict |>
        groupby(__,[:winstart,:winlen,:salience_label,:target_time_label,:sid]) |>
        combine(__,[:correct,:weight] => wmeanish => :correct) |>
        groupby(__,[:winlen,:salience_label,:target_time_label]) |>
        combine(__,:correct => mean) |>
        insertcols!(__,:condition => "object")

    spatial_winlen_means = @_ spatial_classpredict |>
        groupby(__,[:winstart,:winlen,:salience_label,:target_time_label,:sid]) |>
        combine(__,[:correct,:weight] => wmeanish => :correct) |>
        groupby(__,[:winlen,:salience_label,:target_time_label]) |>
        combine(__,:correct => mean) |>
        insertcols!(__,:condition => "spatial")

    best_windows = @_ vcat(object_winlen_means,spatial_winlen_means) |>
        groupby(__,[:salience_label,:target_time_label,:condition]) |>
        combine(__,[:winlen,:correct_mean] =>
            ((len,val) -> len[argmax(val)]) => :winlen)

    best_windows_file = joinpath(paramdir,savename("best_windows_sal_target_time",
        (absolute=use_absolute_features,),"json"))

    @tagsave best_windows_file Dict(
        :data => JSONTables.ObjectTable(Tables.columns(best_windows)),
        :seed => seed
    ) safe=true
end
