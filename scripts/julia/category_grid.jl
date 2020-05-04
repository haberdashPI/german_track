using DrWatson
@quickactivate("german_track")

# we parallelize model parameter optimization with multi-process
# computing; I'd love to use multi-threading, but the current model
# is a python implementaiton and PyCall doesn't support multi-threading
# it's not even clear that'st technically feasible given the python GIL
using Distributed
addprocs(12,exeflags="--project=.")

using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures,
    Dates, Underscores, StatsBase, Random, Printf, ProgressMeter, VegaLite,
    FileIO, StatsBase, RCall, Bootstrap, BangBang, Transducers, PyCall

# local only packages
using Formatting

using ScikitLearn

import GermanTrack: stim_info, speakers, directions, target_times, switch_times

@everywhere begin
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

classdf_file = joinpath(cache_dir(),"data","freqmeans_trial.csv")
if isfile(classdf_file)
    classdf = CSV.read(classdf_file)
else
    # NOTE: in principle this could also be run in parallel (doesn't feel worth it
    # since FFTW doesn't handle multi-threading yet, so we'd have to setup more
    # multi-process parallism)

    classdf = find_powerdiff(
        subjects,groups=[:salience,:target_time,:trial,:sound_index],
        hittypes = ["hit"],
        winlens = 2.0 .^ range(-1,1,length=10),
        winstarts = [0; 2.0 .^ range(-2,1,length=9)])
    CSV.write(classdf_file,classdf)
end

# classdf_file = joinpath(cache_dir(),"data","freqmeans_sal.csv")
# if isfile(classdf_file)
#     classdf = CSV.read(classdf_file)
# else
#     classdf = find_powerdiff(
#         subjects,groups=[:salience],
#         hittypes = ["hit"],
#         winlens = 2.0 .^ range(-1,1,length=10),
#         winstarts = [0; 2.0 .^ range(-2,1,length=9)])
#     CSV.write(classdf_file,classdf)
# end


@everywhere begin
    classdf_file = joinpath(cache_dir(),"data","freqmeans_trial.csv")
    # classdf_file = joinpath(cache_dir(),"data","freqmeans.csv")
    # classdf_file = joinpath(cache_dir(),"data","freqmeans_sal.csv")
    classdf = CSV.read(classdf_file)

    objectdf = @_ classdf |> filter(_.condition in ["global","object"],__)
end

@everywhere begin
    function modelacc(sdf,params)
        # some values of nu may be infeasible, so we have to
        # catch those and return the worst possible fitness
        try
            result = testmodel(sdf,NuSVC(;params...),
                :sound_index,:condition,r"channel",n_folds=3)
            result.correct |> sum
        catch e
            if e isa PyCall.PyError
                return 0
            else
                rethrow(e)
            end
        end
    end
end

vset = @_ objectdf.sound_index |> unique |>
    StatsBase.sample(MersenneTwister(111820),__1,round(Int,0.2length(__1)))
valgroups = @_ objectdf |> filter(_.sound_index ∈ vset,__) |>
    groupby(__, [:winstart,:winlen,:salience])

param_range = (nu=(0.0,0.75),gamma=(-4.0,1.0))
param_by = (nu=identity,gamma=x -> 10^x)
opts = (
    by=param_by,
    MaxFuncEvals = 100,
    PopulationSize = 25,
)

# NOTE: progress bar initial ETA will be a gross overestimate
progress = Progress(opts.MaxFuncEvals,"Optimizing params...")
best_params, fitness = optparams(param_range;opts...) do params
    gr = collect(valgroups)
    correct = dreduce(+,Map(i -> modelacc(valgroups[i],params)),1:length(gr))
    next!(progress)
    N = sum(g -> size(g,1),gr)
    return 1 - correct/N
end
finish!(progress)
best_params = GermanTrack.apply(param_by,best_params)

@everywhere function modelresult((key,sdf))
    result = testmodel(sdf,NuSVC(nu=0.5,gamma=0.04),
        :sid,:condition,r"channel")
    foreach(kv -> result[!,kv[1]] .= kv[2],pairs(key))
    result
end
testgroups = @_ objectdf |> #filter(_.sound_index ∉ vset,__) |>
    groupby(__, [:winstart,:winlen,:salience])
classpredict = dreduce(append!!,Map(modelresult),collect(pairs(testgroups)),
    init=Empty(DataFrame))

subj_means = @_ classpredict |>
    by(__,[:winstart,:winlen,:salience],:correct => mean)

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
        color={:correct_mean,scale={reverse=true,domain=[0.5,0.75],scheme="plasma"}},
        column=:salience)

save(joinpath(dir,"by_trial_svm_allbins.pdf"),pl)

best_high = @_ subj_means |> filter(_.salience == "high",__) |>
    sort(__,:correct_mean,rev=true) |>
    first(__,1)
best_low = @_ subj_means |> filter(_.salience == "low",__) |>
    sort(__,:correct_mean,rev=true) |>
    first(__,1)

best_vals = @_ classpredict |>
    filter((_1.winstart == best_high.winstart[1] &&
            _1.winlen == best_high.winlen[1]) ||
           (_1.winstart == best_low.winstart[1] &&
            _1.winlen == best_low.winlen[1]),__) |>
    by(__,[:winlen,:winstart,:salience],:correct => function(x)
        bs = bootstrap(mean,x,BasicSampling(10_000))
        μ,low,high = 100 .* confint(bs,BasicConfInt(0.683))[1]
        (correct = μ, low = low, high = high)
    end)

best_vals.winlen .= round.(best_vals.winlen,digits=2)
best_vals[!,:window] .= (format.("width = {:1.2f}s, start = {:1.2f}s",
    best_vals.winlen,best_vals.winstart))

pl =
    @vlplot() +
    @vlplot(data=[{}], mark=:rule,
    encoding = {
      y = {datum = 50},
      strokeDash = {value = [2,2]}
    }) +
    (best_vals |>
     @vlplot(x={:window, type=:ordinal, axis={title="Window"}}) +
     @vlplot(mark={:errorbar,filled=true},
            y={"low",scale={zero=false}, axis={title=""},type=:quantitative},
            y2={"high", type=:quantitative}, color=:salience) +
     @vlplot(mark={:point,filled=true},
            y={:correct,scale={zero=false},axis={title="% Correct Classification"}},
            color=:salience))


# TODO: add a dotted line to chance level

save(joinpath(dir, "object_by_trial_best_windows.pdf"),pl)

# use trial based freqmeans below
powerdf_timing = @_ freqmeans_bytime |>
    stack(__, Between(:delta,:gamma),
        variable_name = :freqbin, value_name = :power) |>
    filter(all(!isnan,_.power), __)

powerdiff_timing_df = @_ powerdf_timing |>
    unstack(__, :window_timing, :power) |>
    filter(_.hit != :baseline,__) |>
    filter(_.salience == "low",__) |>
    by(__, [:sid,"hit",:freqbin,:condition,:winstart,:winlen,:channel,:target_time],
        (:before,:after) => sdf ->
            (powerdiff = mean(log.(ε .+ sdf.after) .- log.(ε .+ sdf.before)),))

powerdiff_timing_df[!,"hit"_channel_bin] .=
    categorical(Symbol.(:channel_,powerdiff_timing_df.channel,:_,powerdiff_timing_df.hit,:_,powerdiff_timing_df.freqbin))

classdf_timing = @_ powerdiff_timing_df |>
    unstack(__, [:sid, :condition, :winstart, :winlen, :target_time],
        "hit"_channel_bin, :powerdiff) |>
    filter(all(!ismissing,_[r"channel"]), __) |>
    disallowmissing!(__,r"channel")

objectdf_timing = @_ classdf_timing |> filter(_.condition in [:global,:object],__)


prog = Progress(length(groupby(objectdf_timing,[:winstart,:winlen,:target_time])))
classpredict = by(objectdf_timing, [:winstart,:winlen,:target_time]) do sdf
    result = testmodel(sdf,NuSVC,
        (nu=(0.0,0.75),gamma=(-4.0,1.0)),by=(nu=identity,gamma=x -> 10^x),
        max_evals = 100,:sid,:condition,r"channel")
    next!(prog)
    result[!,:sid] .= sdf.sid
    result
end



## plot raw data
bestwindow_df = @_ powerdiff_df |>
    filter((_1.winstart == best_high.winstart[1] &&
            _1.winlen == best_high.winlen[1]) ||
           (_1.winstart == best_low.winstart[1] &&
            _1.winlen == best_low.winlen[1]),__) |>
    filter(_.hit != :baseline,__) |>
    by(__,[:sid,"hit",:freqbin,:condition,:winlen,:channel,:salience],:powerdiff => mean ∘ skipmissing)

# TODO: compute euclidean difference

bestwindow_df.hit = string.(bestwindow_df.hit)
bestwindow_df.freqbin = string.(bestwindow_df.freqbin)
bestwindow_df.condition = string.(bestwindow_df.condition)
bestwindow_df.salience = string.(bestwindow_df.salience)

R"""

library(ggplot2)
library(dplyr)

plotdf = $bestwindow_df %>% filter(channel == 1) %>%
    mutate(freqbin = factor(freqbin,levels=c('delta','theta','alpha','beta','gamma'),ordered=T))

ggplot(plotdf,aes(x=freqbin,y=powerdiff_function,color=hit,group=hit)) +
    facet_wrap(salience~condition) +
    stat_summary(geom='pointrange',position=position_dodge(width=0.5)) +
    geom_point(size=0.5,alpha=0.5,position=position_jitterdodge(dodge.width=0.3,jitter.width=0.1))

"""

spatialdf = @_ classdf |> filter(_.condition in [:global,:spatial],__)


prog = Progress(length(groupby(spatialdf,[:winstart,:winlen,:salience])))
classpredict = by(spatialdf, [:winstart,:winlen,:salience]) do sdf
    result = testmodel(sdf,NuSVC,
        (nu=(0.0,0.75),gamma=(-4.0,1.0)),by=(nu=identity,gamma=x -> 10^x),
        max_evals = 100,:sid,:condition,r"channel")
    next!(prog)
    result[!,:sid] .= sdf.sid
    result
end

subj_means = @_ classpredict |>
    by(__,[:winstart,:winlen,:salience],:correct => mean)

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
        column=:salience)

save(joinpath(dir,"spatial_svm_allbins.pdf"),pl)

best_high = @_ subj_means |> filter(_.salience == "high",__) |>
    sort(__,:correct_mean,rev=true) |>
    first(__,1)
best_low = @_ subj_means |> filter(_.salience == "low",__) |>
    sort(__,:correct_mean,rev=true) |>
    first(__,1)

best_vals = @_ classpredict |>
    filter((_1.winstart == best_high.winstart[1] &&
            _1.winlen == best_high.winlen[1]) ||
           (_1.winstart == best_low.winstart[1] &&
            _1.winlen == best_low.winlen[1]),__) |>
    by(__,[:winlen,:salience],:correct => function(x)
        bs = bootstrap(mean,x,BasicSampling(10_000))
        μ,low,high = 100 .* confint(bs,BasicConfInt(0.683))[1]
        (correct = μ, low = low, high = high)
    end)
best_vals.winlen .= round.(best_vals.winlen,digits=2)

pl =
    @vlplot() +
    @vlplot(data=[{}], mark=:rule,
    encoding = {
      y = {datum = 50, type=:quantitative},
      strokeDash = {value = [2,2]}
    }) +
    (best_vals |>
     @vlplot(x={:winlen, type=:ordinal, axis={title="Length (s)"}}) +
     @vlplot(mark={:errorbar,filled=true},
            y={"low",scale={zero=true}, axis={title=""},type=:quantitative},
            y2={"high", type=:quantitative}, color={:salience, type=:nominal}) +
     @vlplot(mark={:point,filled=true},
            y={:correct,type=:quantitative,scale={zero=true},axis={title="% Correct Classification"}},
            color={:salience, type=:nominal}))

save(File(format"PDF",joinpath(dir, "spatial_best_windows.pdf")),pl)
