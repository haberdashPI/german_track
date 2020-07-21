# Setup
# =================================================================

using DrWatson
@quickactivate("german_track")
use_cache = true
seed = 072189
use_absolute_features = true
n_winlens = 12
n_folds = 10
n_procs = 6
classifier = :svm_radial

using Distributed

if !(nprocs() > 1) && n_procs > 1
    addprocs(n_procs, exeflags = "--project=.")
end

@everywhere begin
    classifier = Symbol($(string(classifier)))
    n_folds = $n_folds
    seed = $seed

    using EEGCoding,
        GermanTrack,
        DataFrames,
        Statistics,
        Dates,
        Underscores,
        Random,
        ProgressMeter,
        FileIO,
        StatsBase,
        RCall,
        BangBang,
        Transducers,
        PyCall,
        Alert,
        JSON3,
        JSONTables,
        Formatting,
        ScikitLearn,
        Distributions

    import GermanTrack: stim_info
end

@everywhere begin
    @sk_import svm: SVC
end


dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

R"""
library(ggplot2)
library(cowplot)
library(dplyr)
library(Hmisc)
"""

@everywhere begin
    wmeanish(x,w) = iszero(sum(w)) ? 0.0 : mean(coalesce.(x,one(eltype(x))/2),weights(w))
end

# TODO: adapt the below analysis for switch
# Baseline: object & spatial
# =================================================================

# Load SVM parameters
# -----------------------------------------------------------------

paramdir    = processed_datadir("classifier_params")
paramfile   = joinpath(paramdir,savename("hyper-parameters",
    (absolute = use_absolute_features, classifier = classifier),"json"))
best_params = jsontable(open(JSON3.read,paramfile,"r")[:data]) |> DataFrame
if :subjects in propertynames(best_params) # some old files misnamed the sid column
    rename!(best_params,:subjects => :sid)
end

# Compute frequincy bin power
# -----------------------------------------------------------------

paramdir = processed_datadir("classifier_params")
best_windows_file = joinpath(paramdir,savename("best-windows",
    (absolute = use_absolute_features, classifier = classifier), "json"))
best_windows = jsontable(open(JSON3.read,best_windows_file,"r")[:data]) |> DataFrame

best_baseline = @_ best_windows |>
    groupby(__,:condition) |>
    combine(__,:winlen => mean, :winlen => std)
grouped_winlens = groupby(best_baseline, :condition)

spread(scale,npoints)   = x -> spread(x,scale,npoints)
spread(x,scale,npoints) = quantile.(Normal(x,scale/2),range(0.05,0.95,length=npoints))

function best_winlens_for(df)
    best_winlen = df.condition[1] == "global" ? best_baseline :
        grouped_winlens[(condition = df.condition[1],)]
    winlens = reduce(vcat, spread.(best_winlen.winlen_mean, best_winlen.winlen_std,
        n_winlens))
end

classdf_file = joinpath(cache_dir(), "data", savename("freqmeans_baseline",
    (absolute  = use_absolute_features,
     n_winlens = n_winlens),
     "csv"))

if use_cache && isfile(classdf_file) && mtime(classdf_file) > mtime(best_windows_file)
    classdf = CSV.read(classdf_file)
else
    eeg_files = dfhit = @_ readdir(processed_datadir("eeg")) |> filter(occursin(r".h5$",_), __)
    subjects  = Dict(
        sidfor(file) => load_subject(joinpath(processed_datadir("eeg"), file), stim_info,
                            encoding = RawEncoding())
        for file in eeg_files
    )

    events         = @_ mapreduce(_.events,append!!,values(subjects))
    classdf_groups = @_ events |> groupby(__,[:sid,:condition])

    progress = Progress(length(classdf_groups), desc="Computing frequency bins...")
    classdf_groups = @_ events |>
        filter(_.target_present,__) |>
        insertcols!(__,:hit => ishit.(eachrow(__),region = "target")) |>
        filter(_.hit ∈ ["hit", "miss"],__) |>
        groupby(__,[:hit,:sid,:condition])

    classdf = combine(classdf_groups) do sdf
        winlens = best_winlens_for(sdf)

        x = mapreduce(append!!, winlens) do winlen
            si = sdf.sound_index
            result = compute_powerbin_features(subjects[sdf.sid[1]].eeg, sdf, "baseline",
                (len = winlen, start = -winlen), baseline = (mindist = 0.25, minlen = 0.5))
            result[!,:winlen] .= winlen
            result
        end
        next!(progress)

        x
    end
    ProgressMeter.finish!(progress)

    CSV.write(classdf_file, classdf)
    alert("Freqmeans Complete!")
end

# Window Classification
# -----------------------------------------------------------------

classfile = joinpath(cache_dir(), "data", savename("baseline-classify",
    (absolute = use_absolute_features,), "csv"))
if isfile(classfile) && mtime(classfile) > mtime(classdf_file)
    predict = CSV.read(classfile)
else
    @everywhere function modelresult((key,sdf))
        if length(unique(sdf.condition)) >= 2
            params = classifierparams(sdf[1,:], classifier)
            testclassifier(buildmodel(params, classifier, seed),
                data = @_(filter(_.weight > 0, sdf)), y = :condition, X = r"channel",
                crossval = :sid, seed = hash((params, seed)), n_folds = n_folds)
        else
            Empty(DataFrame)
        end
    end

    function classpredict(df, params, condition, variables...)
        testgroups = @_ df |>
            innerjoin(__, params, on=:sid) |>
            groupby(__, [:winlen, variables..., :fold])
        testgroup_pairs = collect(pairs(testgroups))
        predictions = @show progress @distributed (append!!) for key_sdf in testgroup_pairs
            modelresult(key_sdf)
        end

        processed = @_ predictions |>
            groupby(__,[variables..., :sid]) |> #,:before]) |>
            combine(__,[:correct, :weight] => wmeanish => :correct_mean) |>
            insertcols!(__,:condition => condition)

        processed, predictions
    end

    objectdf = @_ classdf |>
        filter(_.condition in ["global", "object"],__)
    object_predict, object_raw = classpredict(objectdf, best_params, "object", :hit)

    spatialdf = @_ classdf |>
        filter(_.condition in ["global", "spatial"],__)
    spatial_predict, spatial_raw = classpredict(spatialdf, best_params, "spatial", :hit)

    predict = vcat(object_predict,spatial_predict)
    CSV.write(classfile, predict)
end

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

# Plots
# -----------------------------------------------------------------

R"""
pos = position_dodge(width=0.8)
pl = ggplot($predict, aes(x = condition, y = correct_mean, fill = hit)) +
    stat_summary(fun.data = 'mean_cl_boot',
        geom = 'bar', width = 0.5, position = pos) +
    stat_summary(fun.data = 'mean_cl_boot',
        geom = 'linerange', fun.args = list(conf.int = 0.682), position = pos) +
    coord_cartesian(ylim = c(0.4, 1)) +
    geom_hline(yintercept = 0.5, linetype = 2)
"""

R"""
ggsave(file.path($dir,"baseline_classify.pdf"),pl,width=6,height=4)
"""
