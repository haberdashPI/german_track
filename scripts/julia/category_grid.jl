using DrWatson
@quickactivate("german_track")

using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures,
    Dates, Underscores, StatsBase, Random, Printf, ProgressMeter, VegaLite,
    FileIO, StatsBase, RCall, Bootstrap

using ScikitLearn
@sk_import svm: (NuSVC, SVC)

import GermanTrack: stim_info, speakers, directions, target_times, switch_times

eeg_files = dfhit = @_ readdir(data_dir()) |> filter(occursin(r".mcca$",_), __)
subjects = Dict(file => load_subject(joinpath(data_dir(), file), stim_info,
                                     encoding = RawEncoding())
    for file in eeg_files)


dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

cachefile = joinpath(cache_dir(),"..","data_cache","freqmeans.bson")
if !isfile(cachefile)
    freqmeans = organize_data_by(
        subjects,groups=[:salience],hittypes = [:hit,:miss,:baseline],
        winlens = 2.0 .^ range(-3,1,length=10),
        winstarts = 2.0 .^ range(-3,1,length=10)) do signal,fs
            result = computebands(signal,fs)
            if @_ all(0 ≈ _,signal)
                result[:,Between(:delta,:gamma)] .= 0
            end
            result
        end
    @save cachefile freqmeans
    alert()
else
    @load cachefile freqmeans
end

powerdf = @_ freqmeans |>
    stack(__, Between(:delta,:gamma),
        variable_name = :freqbin, value_name = :power) |>
    filter(all(!isnan,_.power), __)

ε = max(1e-8,minimum(filter(!iszero,powerdf.power))/2)
powerdiff_df = @_ powerdf |>
    unstack(__, :window_timing, :power) |>
    filter(_.hit != :baseline,__) |>
    by(__, [:sid,:hit,:freqbin,:condition,:winstart,:winlen,:channel,:salience],
        (:before,:after) => sdf ->
            (powerdiff = mean(log.(ε .+ sdf.after) .- log.(ε .+ sdf.before)),))

powerdiff_df[!,:hit_channel_bin] .=
    categorical(Symbol.(:channel_,powerdiff_df.channel,:_,powerdiff_df.hit,:_,powerdiff_df.freqbin))
classdf = @_ powerdiff_df |>
    unstack(__, [:sid, :condition, :winstart, :winlen, :salience],
        :hit_channel_bin, :powerdiff) |>
    filter(all(!ismissing,_[r"channel"]), __) |>
    disallowmissing!(__,r"channel")

objectdf = @_ classdf |> filter(_.condition in [:global,:object],__)

prog = Progress(length(groupby(objectdf,[:winstart,:winlen,:salience])))
classpredict = by(objectdf, [:winstart,:winlen,:salience]) do sdf
    result = testmodel(sdf,NuSVC,
        (nu=(0.0,0.75),gamma=(-4.0,1.0)),by=(nu=identity,gamma=x -> 10^x),
        max_evals = 100,:sid,:condition,r"channel")
    next!(prog)
    result[!,:sid] .= sdf.sid
    result
end

mean(classpredict.correct)

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

best_high = @_ subj_means |> filter(_.salience == :high,__) |>
    sort(__,:correct_mean,rev=true) |>
    first(__,1)
best_low = @_ subj_means |> filter(_.salience == :low,__) |>
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
            y={:low,scale={zero=true}, axis={title=""},type=:quantitative},
            y2={:high, type=:quantitative}, color=:salience) +
     @vlplot(mark={:point,filled=true},
            y={:correct,scale={zero=true},axis={title="% Correct Classification"}},
            color=:salience))


# TODO: add a dotted line to chance level

save(joinpath(dir, "object_best_windows.pdf"),pl)


cachefile = joinpath(cache_dir(),"..","data_cache","freqmeans_bytargettime.bson")
if !isfile(cachefile)
    freqmeans_bytime = organize_data_by(
        subjects,groups=[:salience,:target_time],hittypes = [:hit,:miss,:baseline],
        winlens = 2.0 .^ range(-3,1,length=10),
        winstarts = 2.0 .^ range(-3,1,length=10)) do signal,fs
            result = computebands(signal,fs)
            if @_ all(0 ≈ _,signal)
                result[:,Between(:delta,:gamma)] .= 0
            end
            result
        end
    @save cachefile freqmeans_bytime
    alert()
else
    @load cachefile freqmeans_bytime
end

powerdf_timing = @_ freqmeans_bytime |>
    stack(__, Between(:delta,:gamma),
        variable_name = :freqbin, value_name = :power) |>
    filter(all(!isnan,_.power), __)

powerdiff_timing_df = @_ powerdf_timing |>
    unstack(__, :window_timing, :power) |>
    filter(_.hit != :baseline,__) |>
    filter(_.salience == :low,__) |>
    by(__, [:sid,:hit,:freqbin,:condition,:winstart,:winlen,:channel,:target_time],
        (:before,:after) => sdf ->
            (powerdiff = mean(log.(ε .+ sdf.after) .- log.(ε .+ sdf.before)),))

powerdiff_timing_df[!,:hit_channel_bin] .=
    categorical(Symbol.(:channel_,powerdiff_timing_df.channel,:_,powerdiff_timing_df.hit,:_,powerdiff_timing_df.freqbin))

classdf_timing = @_ powerdiff_timing_df |>
    unstack(__, [:sid, :condition, :winstart, :winlen, :target_time],
        :hit_channel_bin, :powerdiff) |>
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
    by(__,[:sid,:hit,:freqbin,:condition,:winlen,:channel,:salience],:powerdiff => mean ∘ skipmissing)

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

best_high = @_ subj_means |> filter(_.salience == :high,__) |>
    sort(__,:correct_mean,rev=true) |>
    first(__,1)
best_low = @_ subj_means |> filter(_.salience == :low,__) |>
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
            y={:low,scale={zero=true}, axis={title=""},type=:quantitative},
            y2={:high, type=:quantitative}, color={:salience, type=:nominal}) +
     @vlplot(mark={:point,filled=true},
            y={:correct,type=:quantitative,scale={zero=true},axis={title="% Correct Classification"}},
            color={:salience, type=:nominal}))

save(File(format"PDF",joinpath(dir, "spatial_best_windows.pdf")),pl)
