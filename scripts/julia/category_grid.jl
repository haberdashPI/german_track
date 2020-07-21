# Setup
# =================================================================

using DrWatson
@quickactivate("german_track")
use_cache = true
seed = 072189
n_fun_evals = 5_000
test_optimization = false
num_local_procs = 1
num_cluster_procs = 16
use_absolute_features = true
use_slurm = gethostname() == "lcap.cluster"
classifiers = :svm_radial, :svm_linear, :gradient_boosting, :logistic_l1
classifier = classifiers[4]
classifier ∈ classifiers || error("Unexpected classifier $classifier")

using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures,
    Dates, Underscores, Random, Printf, ProgressMeter, VegaLite, FileIO,
    StatsBase, Bootstrap, BangBang, Transducers, PyCall, ScikitLearn, Flux,
    JSON3, JSONTables, Tables, Infiltrator, FileIO, BlackBoxOptim

DrWatson._wsave(file, data::Dict) = open(io -> JSON3.write(io, data), file, "w")

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
    if !(nprocs() > 1) && num_cluster_procs > 1 && !test_optimization
        addprocs(SlurmManager(num_cluster_procs),
            partition = "CPU",
            t = "32:00:00",
            mem = "32G",
            exeflags = "--project=.")
    end
else
    if !(nprocs() > 1) && num_local_procs > 1
        addprocs(num_local_procs, exeflags = "--project=.")
    end
end

@everywhere begin
    using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures,
        Dates, Underscores, Random, Printf, ProgressMeter, VegaLite, FileIO,
        StatsBase, Bootstrap, BangBang, Transducers, PyCall, ScikitLearn, Flux,
        JSON3, JSONTables, Tables, FileIO

    import GermanTrack: stim_info, speakers, directions, target_times, switch_times

    wmeanish(x,w) = iszero(sum(w)) ? 0.0 : mean(coalesce.(x,one(eltype(x))/2),weights(w))
end

@everywhere classifier = Symbol($(string(classifier))) # bug means we can't pass symbol values normally
@everywhere seed = $seed

@everywhere( @sk_import svm: SVC )
@everywhere( @sk_import ensemble: GradientBoostingClassifier )
@everywhere( @sk_import linear_model: LogisticRegression )

if !use_slurm
    dir = joinpath(plotsdir(), string("results_", Date(now())))
    isdir(dir) || mkdir(dir)
end

# is freq means always the same?

# Mean Frequency Bin Analysis
# =================================================================

isdir(processed_datadir("features")) || mkdir(processed_datadir("features"))
classdf_file = joinpath(processed_datadir("features"), savename("freaqmeans",
    (absolute = use_absolute_features,), "csv"))

if use_cache && isfile(classdf_file)
    classdf = CSV.read(classdf_file)
else
    windows = [(len = len, start = start, before = -len)
        for len in 2.0 .^ range(-1, 1, length = 7),
            start in [0; 2.0 .^ range(-2, 2, length = 6)]]
    eeg_files = dfhit = @_ readdir(processed_datadir("eeg")) |>
        filter(occursin(r".h5$", _), __)
    subjects = Dict(
        sidfor(file) => load_subject(
            joinpath(processed_datadir("eeg"), file), stim_info,
            encoding = RawEncoding()
        ) for file in eeg_files)

    events = @_ mapreduce(_.events, append!!, values(subjects))
    classdf_groups = @_ events |>
        filter(_.target_present, __) |>
        filter(ishit(_, region = "target") == "hit", __) |>
        groupby(__, [:salience_label, :target_time_label, :sid, :condition])

    progress = Progress(length(classdf_groups), desc = "Computing frequency bins...")
    classdf = @_ classdf_groups |>
        combine(function(sdf)
            # compute features in each window
            x = mapreduce(append!!, windows) do window
                result = if use_absolute_features
                    compute_powerbin_features(subjects[sdf.sid[1]].eeg, sdf,
                        "target", window)
                else
                    compute_powerdiff_features(subjects[sdf.sid[1]].eeg, sdf,
                        "target", window)
                end
                result[!, :winstart] .= window.start
                result[!, :winlen] .= window.len
                result
            end
            next!(progress)
            x
        end, __)
    ProgressMeter.finish!(progress)
    CSV.write(classdf_file, classdf)

end

# Hyper-parameter Optimization
# =================================================================

objectdf = @_ classdf |> filter(_.condition in ["global", "object"], __)
spatialdf = @_ classdf |> filter(_.condition in ["global", "spatial"], __)

# Function Definitions
# -----------------------------------------------------------------

@everywhere begin
    np = pyimport("numpy")
    inner_n_folds = 10
    # _wmean(x, weight) = (sum(x.*weight) + 1) / (sum(weight) + 2)

    function resultmax(result, conditions...)
        if result isa Vector{<:NamedTuple}
            maxacc = @_ DataFrame(result) |>
                groupby(__, collect(conditions)) |>
                combine(__, :mean => maximum => :max)
            return mean(maxacc.max)#/length(gr)
        else
            @info "Exception: $result"
            return 0.0
        end
    end

    function modelacc((key, sdf), params)
        global classifier
        # some values of nu may be infeasible, so we have to
        # catch those and return the worst possible fitness
        try
            result = testclassifier(buildmodel(params, classifier, seed), data = sdf,
                y = :condition, X = r"channel", crossval = :sid, n_folds = inner_n_folds,
                seed = hash((params, seed)))

            return (
                mean   = wmeanish(result.correct, result.weight),
                weight = sum(result.weight),
                NamedTuple(key)...
            )
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

param_range, param_by = if classifier == :svm_radial
    (C=(-3.0, 3.0), gamma=(-4.0, 1.0)), (C = x -> 10^x, gamma = x -> 10^x)
elseif classifier ∈ (:svm_linear, :logistic_l1)
    (C=(-3.0, 3.0), ), (C = x -> 10^x, )
elseif classifier == :gradient_boosting
    p = (
        max_depth     = (1.0,  5.0),
        n_estimators  = (10.0, 251.0),
        learning_rate = (-3.0, 0.0),
    )
    f = (
        max_depth     = x -> floor(Int,x),
        n_estimators  = x -> floor(Int,x),
        learning_rate = x -> 10^x,
    )
    p, f
else
    error("Unrecognized classifier: $classifier")
end

opts = (
    MaxFuncEvals = test_optimization ? 2 : n_fun_evals,
    FitnessTolerance = 0.03,
    TargetFitness = 0.0,
    # PopulationSize = 25,
)
n_folds = 2

all_opts = (
    SearchRange = collect(values(param_range)),
    NumDimensions = length(param_range),
    TraceMode = :silent,
    opts...
)

# type piracy: awaiting PR acceptance to remove
JSON3.StructTypes.StructType(::Type{<:CategoricalValue{<:String}}) =
    JSON3.StructTypes.StringType()

paramdir = processed_datadir("classifier_params")
isdir(paramdir) || mkdir(paramdir)
paramfile = joinpath(paramdir, savename("hyper-parameters", (
    absolute   = use_absolute_features,
    classifier = classifier,
), "json"))

if test_optimization || use_slurm || !use_cache || !isfile(paramfile)
    progress = Progress(opts.MaxFuncEvals*n_folds, "Optimizing params...")
    let result = Empty(DataFrame)
        for (k, (train, test)) in enumerate(folds(n_folds, objectdf.sid |> unique))
            reducefn = test_optimization ? foldl : dreduce
            Random.seed!(hash((seed, :object, k)))

            optresult = bboptimize(;all_opts...) do params
                tparams_vals = @_ map(_1(_2), param_by, params)
                tparams = NamedTuple{keys(param_by)}(tparams_vals)

                objectgr = @_ objectdf |> filter(_.sid ∈ train, __) |>
                    groupby(__, [:winstart, :winlen, :salience_label,
                        :target_time_label]) |>
                    pairs |> collect

                objectresult = reducefn(append!!,
                    Map(i -> [modelacc(objectgr[i], tparams)]),
                    1:length(objectgr), init = Empty(Vector))

                spatialgr = @_ spatialdf |> filter(_.sid ∈ train, __) |>
                    groupby(__, [:winstart, :winlen, :salience_label,
                        :target_time_label]) |>
                    pairs |> collect

                spatialresult = reducefn(append!!,
                    Map(i -> [modelacc(spatialgr[i], tparams)]),
                    1:length(spatialgr), init = Empty(Vector))

                # spatialresult = dreduce(append!!,
                #     Map(i -> [modelacc(spatialgr[i], params)]),
                #     1:length(spatialgr), init = Empty(Vector))

                next!(progress)

                maxacc = max(
                    resultmax(objectresult, :salience_label, :target_time_label),
                    resultmax(spatialresult, :salience_label, :target_time_label)
                )

                return 1.0 - maxacc
            end

            fold_params, fitness = best_candidate(optresult), best_fitness(optresult)
            fold_params_vals = @_ map(_1(_2), param_by, fold_params)
            fold_params = NamedTuple{keys(param_by)}(fold_params_vals)
            result = append!!(result, DataFrame(sid = test, fold = k, fitness = fitness; fold_params...))
        end

        ProgressMeter.finish!(progress)
        global best_params = result

        # save a reproducible record of the results
        if !test_optimization
            @tagsave paramfile Dict(
                :data => JSONTables.ObjectTable(Tables.columns(best_params)),
                :seed => seed,
                :param_range => param_range,
                :n_folds => n_folds,
                :inner_n_folds => inner_n_folds,
                :optimize_parameters => Dict(k => v for (k, v) in pairs(opts) if k != :by)
            ) safe = true
        end
    end
else
    global best_params = jsontable(open(JSON3.read, paramfile, "r")[:data]) |> DataFrame
    if :subjects in propertynames(best_params) # some old files misnamed the sid column
        rename!(best_params, :subjects => :sid)
    end
end

# Object Classification Results
# =================================================================

if !use_slurm && !test_optimization

    @everywhere function modelresult((key, sdf))
        params = classifierparams(sdf[1,:], classifier)
        testclassifier(buildmodel(params, classifier, seed), data = sdf,
            y = :condition, X = r"channel", crossval = :sid, seed = hash((params, seed)),
            n_folds = inner_n_folds)
    end

    testgroups = @_ objectdf |>
        innerjoin(__, best_params, on = :sid) |>
        groupby(__, [:winstart, :winlen, :salience_label, :target_time_label, :fold])
    object_classpredict = foldl(append!!, Map(modelresult),
        collect(pairs(testgroups)), init = Empty(DataFrame))

    subj_means = @_ object_classpredict |>
        groupby(__, [:winstart, :winlen, :salience_label, :target_time_label, :sid]) |>
        combine(__, [:correct, :weight] => wmeanish => :correct)
    wimeans = @_ subj_means |>
        groupby(__, [:winstart, :winlen, :salience_label, :target_time_label]) |>
        combine(__, :correct => mean)

    sort!(wimeans, order(:correct_mean, rev = true))
    first(wimeans, 6)

    dir = joinpath(plotsdir(), string("results_", Date(now())))
    isdir(dir) || mkdir(dir)

    wimeans.llen = log.(2, wimeans.winlen)
    wimeans.lstart = log.(2, wimeans.winstart)

    pl = wimeans |>
        @vlplot(:rect,
            x = { field = :lstart, bin = {step = 0.573}, },
            y = { field = :llen, bin = {step = 2/9}, },
            color = {
                :correct_mean,
                scale = {reverse = true, domain = [0.5, 1], scheme = "plasma"}
            },
            column = :salience_label,
            row = :target_time_label)

    if use_absolute_features
        save(File(format"PDF",joinpath(dir, "object_grid_absolute_$(classifier).pdf")), pl)
    else
        save(File(format"PDF",joinpath(dir, "object_grid_$(classifier).pdf")), pl)
    end
end

# Classifciation Results: Global v Spattial
# =================================================================

if !use_slurm && !test_optimization

    @everywhere function modelresult((key, sdf))
        params = classifierparams(sdf[1,:], classifier)
        if length(unique(sdf.condition)) == 1
            @info "Skipping data with one class: $(first(sdf, 1))"
            Empty(DataFrame)
        else
            testclassifier(buildmodel(params, classifier, seed), data = sdf,
                y = :condition, X = r"channel", crossval = :sid,
                seed = hash((params, seed)), n_folds = inner_n_folds)
        end
    end

    testgroups = @_ spatialdf |>
        innerjoin(__, best_params, on = :sid) |>
        groupby(__, [:winstart, :winlen, :salience_label, :target_time_label, :fold])
    spatial_classpredict = foldl(append!!, Map(modelresult),
        collect(pairs(testgroups)), init = Empty(DataFrame))

    subj_means = @_ spatial_classpredict |>
        groupby(__, [:winstart, :winlen, :salience_label, :target_time_label, :sid]) |>
        combine(__, [:correct, :weight] => wmeanish => :correct)
    wimeans = @_ subj_means |>
        groupby(__, [:winstart, :winlen, :salience_label, :target_time_label]) |>
        combine(__, :correct => mean)

    sort!(wimeans, order(:correct_mean, rev = true))
    first(wimeans, 6)

    dir = joinpath(plotsdir(), string("results_", Date(now())))
    isdir(dir) || mkdir(dir)

    wimeans.llen = log.(2, wimeans.winlen)
    wimeans.lstart = log.(2, wimeans.winstart)

    pl = wimeans |>
        @vlplot(:rect,
            x = {
                field = :lstart,
                bin = {step = 0.573},
            },
            y = {
                field = :llen,
                bin = {step = 2/9},
            },
            color = {:correct_mean,
                scale = {reverse = true, domain = [0.5, 1], scheme = "plasma"}},
            column = :salience_label, row = :target_time_label)


    if use_absolute_features
        save(File(format"PDF",joinpath(dir, "spatial_grid_absolute_$classifier.pdf")), pl)
    else
        save(File(format"PDF",joinpath(dir, "spatial_grid_$classifier.pdf")), pl)
    end

end

# Find Best Window Length
# =================================================================

@static if !use_slurm && !test_optimization

    object_winlen_means = @_ object_classpredict |>
        groupby(__, [:winstart, :winlen, :salience_label, :target_time_label, :sid]) |>
        combine(__, [:correct, :weight] => wmeanish => :correct) |>
        groupby(__, [:winlen, :salience_label, :target_time_label]) |>
        combine(__, :correct => mean) |>
        insertcols!(__, :condition => "object")

    spatial_winlen_means = @_ spatial_classpredict |>
        groupby(__, [:winstart, :winlen, :salience_label, :target_time_label, :sid]) |>
        combine(__, [:correct, :weight] => wmeanish => :correct) |>
        groupby(__, [:winlen, :salience_label, :target_time_label]) |>
        combine(__, :correct => mean) |>
        insertcols!(__, :condition => "spatial")

    best_windows = @_ vcat(object_winlen_means, spatial_winlen_means) |>
        groupby(__, [:salience_label, :target_time_label]) |>
        combine(__, [:winlen, :correct_mean] =>
            ((len, val) -> len[argmax(val)]) => :winlen)

    best_windows_file = joinpath(paramdir, savename("best-windows",
        (absolute = use_absolute_features, classifier = classifier), "json"))

    @tagsave best_windows_file Dict(
        :data => JSONTables.ObjectTable(Tables.columns(best_windows)),
        :seed => seed
    ) safe = true
end
