# Setup
# =================================================================

using DrWatson
@quickactivate("german_track")
use_cache = true
seed = 072189
use_absolute_features = true
n_winlens = 6
n_winstarts = 64
n_folds = 10

using EEGCoding,
    GermanTrack,
    DataFrames,
    Statistics,
    DataStructures,
    Dates,
    Underscores,
    StatsBase,
    Random,
    Printf,
    ProgressMeter,
    VegaLite,
    FileIO,
    StatsBase,
    RCall,
    Bootstrap,
    BangBang,
    Transducers,
    PyCall,
    Distributions,
    Alert,
    JSON3,
    JSONTables,
    Formatting,
    ScikitLearn,
    Distributions

import GermanTrack: stim_info, speakers, directions, target_times, switch_times

@sk_import svm: SVC

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

R"""
library(ggplot2)
library(cowplot)
library(dplyr)
library(Hmisc)
"""

wmeanish(x,w) = iszero(sum(w)) ? 0.0 : mean(coalesce.(x,one(eltype(x))/2),weights(w))

# Freqmeans Analysis
# =================================================================

paramdir = processed_datadir("svm_params")
best_windows_file = joinpath(paramdir,savename("best-windows",
    (absolute = use_absolute_features,), "json"))
best_windows = jsontable(open(JSON3.read,best_windows_file,"r")[:data]) |> DataFrame

# we use the best window length (determined by average performance across all window starts)
# per condition, salience and target_time, and some spread of values near those best window
# lengths
spread(scale,npoints)   = x -> spread(x,scale,npoints)
spread(x,scale,npoints) = quantile.(Normal(x,scale/2),range(0.05,0.95,length=npoints))
windowtypes = ["target", "baseline"]

grouped_winlens = groupby(best_windows,[:salience_label,:target_time_label,:condition])
function best_windows_for(df)
    best_winlen = if df.condition[1] == "global"
        vcat(grouped_winlens[(
            salience_label    = df.salience_label[1],
            target_time_label = df.target_time_label[1],
            condition         = "object"
        )].winlen,
        grouped_winlens[(
            salience_label    = df.salience_label[1],
            target_time_label = df.target_time_label[1],
            condition         = "spatial"
        )].winlen)
    else
        grouped_winlens[(
            salience_label    = df.salience_label[1],
            target_time_label = df.target_time_label[1],
            condition   = df.condition[1]
        )].winlen
    end
    winlens   = reduce(vcat,spread.(best_winlen,0.5,n_winlens))
    winstarts =  range(0,3,length=n_winstarts)

    Iterators.flatten(
        [Iterators.product(winstarts, winlens,["target"]),
         zip(.-winlens, winlens, fill("baseline", length(winlens)))])

end

classdf_file = joinpath(cache_dir(),"data",
    savename("freqmeans_timeline",
        (absolute    = use_absolute_features,
         n_winlens   = n_winlens,
         n_winstarts = n_winstarts),
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
    classdf_groups = @_ events |>
        filter(_.target_present,__) |>
        insertcols!(__,:hit => ishit.(eachrow(__),region = "target")) |>
        filter(_.hit ∈ ["hit", "miss"],__) |>
        groupby(__,[:hit,:salience_label,:target_time_label,:sid,:condition])

    progress = Progress(length(classdf_groups),desc="Computing frequency bins...")
    classdf  = @_ classdf_groups |>
        combine(function(sdf)
            # setup the windows
            windows = best_windows_for(sdf)

            # compute features in each window
            x = mapreduce(append!!,windows) do (start,len,type)
                result = if use_absolute_features
                    compute_powerbin_features(subjects[sdf.sid[1]].eeg,sdf,"target",
                        (len = len, start = start))
                else
                    compute_powerdiff_features(subjects[sdf.sid[1]].eeg,sdf,"target",
                        (len = len, start = start, before = -len))
                end
                result[!,:winstart] .= start
                result[!,:winlen] .= len
                result[!,:wintype] .= type
                result
            end
            next!(progress)
            x
        end,__)
    ProgressMeter.finish!(progress)

    CSV.write(classdf_file,classdf)
    alert("Freqmeans Complete!")
end

paramdir    = processed_datadir("svm_params")
paramfile   = joinpath(paramdir,savename("hyper-parameters",
    (absolute=use_absolute_features,),"json"))
best_params = jsontable(open(JSON3.read,paramfile,"r")[:data]) |> DataFrame
if :subjects in propertynames(best_params) # some old files misnamed the sid column
    rename!(best_params,:subjects => :sid)
end

# Timelines
# =================================================================

classfile = joinpath(paramdir, savename("timeline-classify",
    (absolute = use_absolute_features,), "csv"))

if isfile(classfile) && mtime(classfile) > mtime(classdf_file)
    predict = CSV.read(classfile)
else
    function modelresult((key,sdf))
        if length(unique(sdf.condition)) >= 2
            params = (C = key[:C], gamma = key[:gamma])
            testclassifier(SVC(;params...),
                data=@_(filter(_.weight > 0,sdf)),y=:condition,X=r"channel",
                crossval=:sid, seed=hash((params,seed)), n_folds=n_folds)
        else
            # in the case where there is one condition, this means that the selected window
            # length has a condition for global but not the second category (object or spatial)
            # this is an indication that the window length is present for use with a category
            # other than the one currently being evaluated and can safely be ignored

            Empty(DataFrame)
        end
    end

    function classpredict(df, params, condition, variables...)
        testgroups = @_ df |>
            innerjoin(__, params, on=:sid) |>
            groupby(__, [:winstart,:winlen, :wintype, variables...,:C,:gamma])
        predictions = foldl(append!!, Map(modelresult),
            collect(pairs(testgroups)), init=Empty(DataFrame))

        processed = @_ predictions |>
            groupby(__,[:winstart, :wintype, variables...,:sid]) |> #,:before]) |>
            combine(__,[:correct,:weight] => wmeanish => :correct_mean) |>
            insertcols!(__,:condition => condition)

        processed, predictions
    end

    objectdf = @_ classdf |>
        filter(_.condition in ["global","object"],__)
    object_predict, object_raw = classpredict(objectdf, best_params, "object", :hit, :salience_label,
        :target_time_label)

    spatialdf = @_ classdf |>
        filter(_.condition in ["global","spatial"],__)
    spatial_predict, spatial_raw = classpredict(spatialdf, best_params, "spatial", :hit,
        :salience_label, :target_time_label)

    predict = vcat(object_predict,spatial_predict)
    CSV.write(classfile, predict)
end

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

# Timeline across salience x target time
# -----------------------------------------------------------------

band = @_ predict |>
    filter(_.hit == "hit",__) |>
    filter(_.wintype != "baseline",__) |>
    groupby(__,[:winstart,:salience_label,:target_time_label,:condition]) |> #,:before]) |>
    combine(:correct_mean => function(correct)
        bs = bootstrap(mean,correct,BasicSampling(10_000))
        μ,low,high = 100 .* confint(bs,BasicConfInt(0.682))[1]
        (correct = μ, low = low, high = high)
    end,__) #|>
    # transform!(__,[:salience,:before] =>
    #     ((x,y) -> string.(x,"_",y)) => :salience_for)

R"""
pl = ggplot($band,
        aes(    x = winstart,
                y = correct,
            color = interaction(salience_label, target_time_label))) +
    geom_ribbon(
        alpha = 0.4,
        aes( ymin = low,
             ymax = high,
             fill = interaction(salience_label, target_time_label),
            color = NULL)) +
    geom_line() +
    facet_grid(~condition) +
    geom_abline(slope = 0, intercept = 50, linetype = 2) +
    guides(fill  = guide_legend(title = "Salience x Target time"),
           color = guide_legend(title = "Salience x Target time")) +
    scale_fill_brewer( palette = 'Paired', direction = -1) +
    scale_color_brewer(palette = 'Paired', direction = -1) +
    coord_cartesian(ylim = c(40, 100))
pl
"""

R"""
ggsave(file.path($dir,"object_salience_timeline.pdf"),pl,width=11,height=8)
"""

# Timeline across salience x target time - baseline
# -----------------------------------------------------------------

winstartish(wintype,winstart) = wintype == "baseline" ? -1.0 : winstart
baseremove = @_ predict |>
    transform!(__,[:wintype,:winstart] => ByRow(winstartish) => :winstartish) |>
    unstack(__,[:salience_label,:target_time_label,:condition,:sid,:hit],:winstartish,:correct_mean,
        renamecols = x -> Symbol("start",x))
baseremove[:,r"start[0-9.]"] .-= Array(baseremove[:,r"start-1.0"])

predict_baseline = @_ baseremove[:,Not(r"start-1.0")] |>
    stack(__,All(r"start"),[:salience_label,:target_time_label,:condition,:sid,:hit],
        variable_name=:winstart, value_name=:correct_mean) |>
    transform!(__,:winstart =>
        (x -> @.(parse(Float64, getindex(match(r"start([0-9.]+)",string(x)),1)))) =>
        :winstart)

band = @_ predict_baseline |>
    filter(_.hit == "hit",__) |>
    groupby(__,[:winstart,:salience_label,:target_time_label,:condition]) |> #,:before]) |>
    combine(:correct_mean => function(correct)
        bs = bootstrap(mean,collect(skipmissing(correct)),BasicSampling(10_000))
        μ,low,high = 100 .* confint(bs,BasicConfInt(0.682))[1]
        (correct = μ, low = low, high = high)
    end,__) #|>
    # transform!(__,[:salience,:before] =>
    #     ((x,y) -> string.(x,"_",y)) => :salience_for)

R"""
pl = ggplot($band,
        aes(    x = winstart,
                y = correct,
            color = interaction(salience_label, target_time_label))) +
    geom_ribbon(
        alpha = 0.4,
        aes( ymin = low,
             ymax = high,
             fill = interaction(salience_label, target_time_label),
            color = NULL)) +
    geom_line() +
    facet_grid(~condition) +
    geom_abline(slope = 0, intercept = 0, linetype = 2) +
    guides(fill  = guide_legend(title = "Salience x Target time"),
           color = guide_legend(title = "Salience x Target time")) +
    scale_fill_brewer( palette = 'Paired', direction = -1) +
    scale_color_brewer(palette = 'Paired', direction = -1)
    # coord_cartesian(ylim = c(40, 100))
pl
"""

R"""
ggsave(file.path($dir,"object_salience_timeline_baseline.pdf"),pl,width=11,height=8)
"""

# Timeline across salineece x target-time - miss
# -----------------------------------------------------------------

predict_miss = @_ predict |>
    filter(_.winstart ≥ 0,__) |>
    unstack(__,[:salience_label,:target_time_label,:condition,:sid,:winstart],
        :hit,:correct_mean) |>
    transform!(__,[:hit,:miss] => (-) => :correct_mean)

band = @_ predict_miss |>
    groupby(__,[:winstart,:salience_label,:target_time_label,:condition]) |> #,:before]) |>
    combine(:correct_mean => function(correct)
        bs = bootstrap(mean,collect(skipmissing(correct)),BasicSampling(10_000))
        μ,low,high = 100 .* confint(bs,BasicConfInt(0.682))[1]
        (correct = μ, low = low, high = high)
    end,__) #|>
    # transform!(__,[:salience,:before] =>
    #     ((x,y) -> string.(x,"_",y)) => :salience_for)

R"""
pl = ggplot($band,aes(x=winstart,y=correct,
        color=interaction(salience_label,target_time_label))) +
    geom_ribbon(aes(ymin=low,ymax=high,
        fill=interaction(salience_label,target_time_label),color=NULL),alpha=0.4) +
    geom_line() + facet_grid(~condition) +
    geom_abline(slope=0,intercept=0,linetype=2) +
    guides(fill = guide_legend(title="Salience x Target time"),
           color = guide_legend(title="Salience x Target time")) +
    scale_fill_brewer( palette = 'Paired', direction = -1) +
    scale_color_brewer(palette = 'Paired', direction = -1)
    ylab('(Hit - Baseline) - (Miss - Baseline)')
    # coord_cartesian(ylim=c(40,100))
pl
"""

R"""
ggsave(file.path($dir,"object_salience_timeline_miss.pdf"),pl,width=11,height=8)
"""

# Timeline across salineece x target-time - baseline - miss
# -----------------------------------------------------------------

predict_base_miss = @_ predict_baseline |>
    unstack(__,[:salience_label,:target_time_label,:condition,:sid,:winstart],
        :hit,:correct_mean) |>
    transform!(__,[:hit,:miss] => (-) => :correct_mean)


band = @_ predict_base_miss |>
    groupby(__,[:winstart,:salience_label,:target_time_label,:condition]) |> #,:before]) |>
    combine(:correct_mean => function(correct)
        bs = bootstrap(mean,collect(skipmissing(correct)),BasicSampling(10_000))
        μ,low,high = 100 .* confint(bs,BasicConfInt(0.682))[1]
        (correct = μ, low = low, high = high)
    end,__) #|>
    # transform!(__,[:salience,:before] =>
    #     ((x,y) -> string.(x,"_",y)) => :salience_for)

R"""
pl = ggplot($band,aes(x=winstart,y=correct,
        color=interaction(salience_label,target_time_label))) +
    geom_ribbon(aes(ymin=low,ymax=high,
        fill=interaction(salience_label,target_time_label),color=NULL),alpha=0.4) +
    geom_line() + facet_grid(~condition) +
    geom_abline(slope=0,intercept=0,linetype=2) +
    guides(fill = guide_legend(title="Salience x Target time"),
           color = guide_legend(title="Salience x Target time")) +
    scale_fill_brewer( palette = 'Paired', direction = -1) +
    scale_color_brewer(palette = 'Paired', direction = -1)
    ylab('(Hit - Baseline) - (Miss - Baseline)')
    # coord_cartesian(ylim=c(40,100))
pl
"""

R"""
ggsave(file.path($dir,"object_salience_timeline_miss_base.pdf"),pl,width=11,height=8)
"""

# Timeline across salience
# -----------------------------------------------------------------

band = @_ predict |>
    filter(_.wintype != "baseline",__) |>
    groupby(__, [:winstart, :salience_label, :condition, :sid]) |> #, :before]) |>
    combine(__, :correct_mean => mean => :correct_mean) |>
    groupby(__, [:winstart, :salience_label, :condition]) |> #, :before]) |>
    combine(:correct_mean => function(correct)
        bs = bootstrap(mean, correct, BasicSampling(10_000))
        μ, low, high = 100 .* confint(bs, BasicConfInt(0.682))[1]
        (correct = μ, low = low, high = high)
    end, __) #|>

R"""
pl = ggplot($band, aes(x = winstart, y = correct, color = salience_label)) +
    geom_ribbon(
            alpha = 0.4,
            aes(ymin  = low,
                ymax  = high,
                fill  = salience_label,
                color = NULL)) +
    geom_line() + facet_grid(~condition) +
    scale_color_brewer(palette = 'Set1') +
    scale_fill_brewer( palette = 'Set1') +
    geom_abline(slope = 0, intercept = 50, linetype = 2) +
    coord_cartesian(ylim = c(40, 100))
pl
"""

R"""
ggsave(file.path($dir,"object_salience.pdf"),pl,width=11,height=8)
"""

R"""
pl = ggplot($band, aes(x = winstart, y = correct, color = salience_label)) +
    geom_ribbon(
            alpha = 0.4,
            aes(ymin  = low,
                ymax  = high,
                fill  = salience_label,
                color = NULL)) +
    geom_line() + facet_grid(~condition) +
    scale_color_brewer(palette = 'Set1') +
    scale_fill_brewer( palette = 'Set1') +
    geom_abline(slope = 0, intercept = 50, linetype = 2) +
    coord_cartesian(ylim = c(40, 70))
pl
"""

R"""
ggsave(file.path($dir,"object_salience.pdf"),pl,width=11,height=8)
"""

# Timeline across salience - baseline
# -----------------------------------------------------------------

band = @_ predict_baseline |>
    groupby(__, [:winstart, :salience_label, :condition, :sid]) |> #, :before]) |>
    combine(__, :correct_mean => mean => :correct_mean) |>
    groupby(__, [:winstart, :salience_label, :condition]) |> #, :before]) |>
    combine(:correct_mean => function(correct)
        bs = bootstrap(mean, collect(skipmissing(correct)), BasicSampling(10_000))
        μ, low, high = 100 .* confint(bs, BasicConfInt(0.682))[1]
        (correct = μ, low = low, high = high)
    end, __) #|>

R"""
pl = ggplot($band, aes(x = winstart, y = correct, color = salience_label)) +
    geom_ribbon(
            alpha = 0.4,
            aes(ymin  = low,
                ymax  = high,
                fill  = salience_label,
                color = NULL)) +
    geom_line() + facet_grid(~condition) +
    scale_color_brewer(palette = 'Set1') +
    scale_fill_brewer( palette = 'Set1') +
    geom_abline(slope = 0, intercept = 0, linetype = 2)
    # coord_cartesian(ylim = c(40, 100))
pl
"""

R"""
ggsave(file.path($dir,"object_salience_baseline.pdf"),pl,width=11,height=8)
"""

# Timeline across salineece - miss
# -----------------------------------------------------------------

band = @_ predict_miss |>
    groupby(__,[:winstart,:salience_label,:condition]) |> #,:before]) |>
    combine(:correct_mean => function(correct)
        bs = bootstrap(mean,collect(skipmissing(correct)),BasicSampling(10_000))
        μ,low,high = 100 .* confint(bs,BasicConfInt(0.682))[1]
        (correct = μ, low = low, high = high)
    end,__) #|>
    # transform!(__,[:salience,:before] =>
    #     ((x,y) -> string.(x,"_",y)) => :salience_for)

R"""
pl = ggplot($band, aes(x = winstart, y = correct, color = salience_label)) +
    geom_ribbon(
            alpha = 0.4,
            aes(ymin  = low,
                ymax  = high,
                fill  = salience_label,
                color = NULL)) +
    geom_line() + facet_grid(~condition) +
    scale_color_brewer(palette = 'Set1') +
    scale_fill_brewer( palette = 'Set1') +
    geom_abline(slope = 0, intercept = 0, linetype = 2)
    # coord_cartesian(ylim = c(40, 100))
pl
"""

R"""
ggsave(file.path($dir,"object_salience_miss.pdf"),pl,width=11,height=8)
"""

# Timeline across salience - baseline - miss
# -----------------------------------------------------------------

band = @_ predict_base_miss |>
    groupby(__,[:winstart,:salience_label,:condition]) |> #,:before]) |>
    combine(:correct_mean => function(correct)
        bs = bootstrap(mean,collect(skipmissing(correct)),BasicSampling(10_000))
        μ,low,high = 100 .* confint(bs,BasicConfInt(0.682))[1]
        (correct = μ, low = low, high = high)
    end,__) #|>
    # transform!(__,[:salience,:before] =>
    #     ((x,y) -> string.(x,"_",y)) => :salience_for)

R"""
pl = ggplot($band,aes(x=winstart,y=correct,
        color=salience_label)) +
    geom_ribbon(aes(ymin=low,ymax=high,
        fill=salience_label,color=NULL),alpha=0.4) +
    geom_line() + facet_grid(~condition) +
    geom_abline(slope=0,intercept=0,linetype=2) +
    guides(fill = guide_legend(title="Salience"),
           color = guide_legend(title="Salience")) +
    scale_fill_brewer(palette='Set1') +
    scale_color_brewer(palette='Set1') +
    ylab('(Hit - Baseline) - (Miss - Baseline)')
    # coord_cartesian(ylim=c(40,100))
pl
"""

R"""
ggsave(file.path($dir,"object_salience_miss_base.pdf"),pl,width=11,height=8)
"""

# Timeline across target time
# -----------------------------------------------------------------

band = @_ predict |>
    # filter(_.before == "zero", __) |>
    filter(_.wintype != "baseline",__) |>
    groupby(__, [:winstart, :target_time_label, :condition, :sid]) |> #, :before]) |>
    combine(__, :correct_mean => mean => :correct_mean) |>
    groupby(__, [:winstart, :target_time_label, :condition]) |> #, :before]) |>
    combine(:correct_mean => function(correct)
        bs = bootstrap(mean, correct, BasicSampling(10_000))
        μ, low, high = 100 .* confint(bs, BasicConfInt(0.682))[1]
        (correct = μ, low = low, high = high)
    end, __) #|>
    # transform!(__, [:salience, :before] =>
    #     ((x, y) -> string.(x, "_", y)) => :salience_for)

R"""
pl = ggplot($band, aes(x = winstart, y = correct, color = target_time_label)) +
    geom_ribbon(
            alpha = 0.4,
            aes(ymin  = low,
                ymax  = high,
                fill  = target_time_label,
                color = NULL)) +
    geom_line() + facet_grid(.~condition) +
    scale_color_brewer(palette = 'Set2') +
    scale_fill_brewer( palette = 'Set2') +
    geom_abline(slope = 0, intercept = 50, linetype = 2) +
    coord_cartesian(ylim = c(40, 100))
pl
"""

R"""
ggsave(file.path($dir, "object_target_time.pdf"), pl, width = 11, height = 8)
"""

# Timeline across target time - baseline
# -----------------------------------------------------------------

band = @_ predict_baseline |>
    # filter(_.before == "zero", __) |>
    groupby(__, [:winstart, :target_time_label, :condition, :sid]) |> #, :before]) |>
    combine(__, :correct_mean => mean => :correct_mean) |>
    groupby(__, [:winstart, :target_time_label, :condition]) |> #, :before]) |>
    combine(:correct_mean => function(correct)
        bs = bootstrap(mean, collect(skipmissing(correct)), BasicSampling(10_000))
        μ, low, high = 100 .* confint(bs, BasicConfInt(0.682))[1]
        (correct = μ, low = low, high = high)
    end, __) #|>
    # transform!(__, [:salience, :before] =>
    #     ((x, y) -> string.(x, "_", y)) => :salience_for)

R"""
pl = ggplot($band, aes(x = winstart, y = correct, color = target_time_label)) +
    geom_ribbon(
            alpha = 0.4,
            aes(ymin  = low,
                ymax  = high,
                fill  = target_time_label,
                color = NULL)) +
    geom_line() + facet_grid(.~condition) +
    scale_color_brewer(palette = 'Set2') +
    scale_fill_brewer( palette = 'Set2') +
    geom_abline(slope = 0, intercept = 0, linetype = 2)
    # coord_cartesian(ylim = c(40, 100))
pl
"""

R"""
ggsave(file.path($dir, "object_target_time_baseline.pdf"), pl, width = 11, height = 8)
"""

# Timeline across salineece - miss
# -----------------------------------------------------------------

band = @_ predict_miss |>
    groupby(__,[:winstart,:target_time_label,:condition]) |> #,:before]) |>
    combine(:correct_mean => function(correct)
        bs = bootstrap(mean,collect(skipmissing(correct)),BasicSampling(10_000))
        μ,low,high = 100 .* confint(bs,BasicConfInt(0.682))[1]
        (correct = μ, low = low, high = high)
    end,__) #|>
    # transform!(__,[:salience,:before] =>
    #     ((x,y) -> string.(x,"_",y)) => :target_time_for)

R"""
pl = ggplot($band, aes(x = winstart, y = correct, color = target_time_label)) +
    geom_ribbon(
            alpha = 0.4,
            aes(ymin  = low,
                ymax  = high,
                fill  = target_time_label,
                color = NULL)) +
    geom_line() + facet_grid(~condition) +
    scale_color_brewer(palette = 'Set1') +
    scale_fill_brewer( palette = 'Set1') +
    geom_abline(slope = 0, intercept = 0, linetype = 2)
    # coord_cartesian(ylim = c(40, 100))
pl
"""

R"""
ggsave(file.path($dir,"object_target_time_miss.pdf"),pl,width=11,height=8)
"""

# Timeline across target time - baseline - miss
# -----------------------------------------------------------------

band = @_ predict_base_miss |>
    groupby(__,[:winstart,:target_time_label,:condition]) |> #,:before]) |>
    combine(:correct_mean => function(correct)
        bs = bootstrap(mean,collect(skipmissing(correct)),BasicSampling(10_000))
        μ,low,high = 100 .* confint(bs,BasicConfInt(0.682))[1]
        (correct = μ, low = low, high = high)
    end,__) #|>
    # transform!(__,[:salience,:before] =>
    #     ((x,y) -> string.(x,"_",y)) => :salience_for)

R"""
pl = ggplot($band,aes(x=winstart,y=correct,
        color=target_time_label)) +
    geom_ribbon(aes(ymin=low,ymax=high,
        fill=target_time_label,color=NULL),alpha=0.4) +
    geom_line() + facet_grid(~condition) +
    geom_abline(slope=0,intercept=0,linetype=2) +
    guides(fill = guide_legend(title="Target Time"),
           color = guide_legend(title="Target Time")) +
    scale_fill_brewer(palette='Set2') +
    scale_color_brewer(palette='Set2') +
    ylab('(Hit - Baseline) - (Miss - Baseline)')
    # coord_cartesian(ylim=c(40,100))
pl
"""

R"""
ggsave(file.path($dir,"object_target_time_miss_base.pdf"),pl,width=11,height=8)
"""
