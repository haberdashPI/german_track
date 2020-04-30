using DrWatson
@quickactivate("german_track")

using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures,
    Dates, Underscores, StatsBase, Random, Printf, ProgressMeter, VegaLite,
    FileIO, StatsBase, RCall, Bootstrap, BangBang, Transducers

using ScikitLearn
@sk_import svm: (NuSVC, SVC)

import GermanTrack: stim_info, speakers, directions, target_times, switch_times

eeg_files = dfhit = @_ readdir(data_dir()) |> filter(occursin(r".mcca$",_), __)
subjects = Dict(file => load_subject(joinpath(data_dir(), file), stim_info,
                                     encoding = RawEncoding())
    for file in eeg_files)


dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

# TODO: try running each window separatley and storing the
# results, rather than storing all versions of the data

classdf = organize_data_by(
    subjects,groups=[:salience,:target_time,:trial,:sound_index],
    hittypes = ["hit"],
    winlens = 2.0 .^ range(-1,1,length=10),
    winstarts = [0; 2.0 .^ range(-2,1,length=9)]) do signal,fs

        freqdf = computebands(signal,fs)
        if @_ all(0 ≈ _,signal)
            freqdf[:,Between(:delta,:gamma)] .= 0
        end

        powerdf = @_ freqdf |>
            stack(__, Between(:delta,:gamma),
                variable_name = :freqbin, value_name = :power) |>
            filter(all(!isnan,_.power), __) |>
            unstack(__, :window_timing, :power)

        ε = 1e-8
        logdiff(x,y) = log.(ε .+ x) .- log.(ε .+ y)
        powerdf[!,:powerdiff] = logdiff(powerdf.after,powerdf.before)
        powerdf[!,:channel_bin] .=
            categorical(string.("channel_",powerdf.channel,"_",powerdf.freqbin))

        classdf = @_ powerdf |>
            unstack(__, [:sid, :trial, :condition, :winstart, :winlen, :salience, :target_time],
                :channel_bin, :powerdiff) |>
            filter(all(!ismissing,_[r"channel"]), __) |>
            disallowmissing!(__,r"channel")

        objectdf = @_ classdf |> filter(_.condition in [:global,:object],__)

        # TODO: in progress, this was what was outside this
        # inner function, we need to change it so
        # the optimization occurs outside of this inner function

        param_range = (nu=(0.0,0.75),gamma=(-4.0,1.0))
        param_by = (nu=identity,gamma=x -> 10^x)
        groups = groupby(objectdf, [:winstart,:winlen,:salience])
        function modelacc(sdf,params,progress)
            result = testmodel(sdf,NuSVC(;params...),:trial,:condition,r"channel")
            next!(progress)

            # TODO: here, this result needs to be returned by
            # the inner function
            # and it will be used to define modelacc
            result.correct |> sum, length(result.correct)
        end

        progress = Progress(100*length(groups))
        best_params, fitness = optparams(param_range,by=param_by,max_evals = 100) do params
            correct,N = foldl((x,y) -> x.+y,Map(x -> modelacc(x,params,progress)),
                collect(groups),init=(0,0))
            1 - correct/N
        end
        finish!(progress)
    end
end

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
        color={:correct_mean,scale={reverse=true,domain=[0.5,1],scheme="plasma"}},
        column=:salience)

save(joinpath(dir,"object_svm_allbins.pdf"),pl)

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
      y = {datum = 50},
      strokeDash = {value = [2,2]}
    }) +
    (best_vals |>
     @vlplot(x={:winlen, type=:ordinal, axis={title="Length (s)"}}) +
     @vlplot(mark={:errorbar,filled=true},
            y={"low",scale={zero=true}, axis={title=""},type=:quantitative},
            y2={"high", type=:quantitative}, color=:salience) +
     @vlplot(mark={:point,filled=true},
            y={:correct,scale={zero=true},axis={title="% Correct Classification"}},
            color=:salience))


# TODO: add a dotted line to chance level

save(joinpath(dir, "object_best_windows.pdf"),pl)

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
