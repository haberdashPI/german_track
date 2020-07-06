# Setup
# =================================================================

using DrWatson
@quickactivate("german_track")
use_cache = true
seed = 072189
use_absolute_features = true
n_winlens = 6
n_winstarts = 64

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
best_windows_file = joinpath(paramdir,savename("hyper-parameters",
    (absolute    = use_absolute_features,), "json"))
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
if use_cache && isfile(classdf_file)
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
    (absolute=use_absolute_features,n_folds=3),"json"))
best_params = jsontable(open(JSON3.read,paramfile,"r")[:data]) |> DataFrame
if :subjects in propertynames(best_params) # some old files misnamed the sid column
    rename!(best_params,:subjects => :sid)
end

# Timelines
# =================================================================

function modelresult((key,sdf))
    if length(unique(sdf.condition)) >= 2
        params = (C = key[:C], gamma = key[:gamma])
        testclassifier(SVC(;params...),
            data=@_(filter(_.weight > 0,sdf)),y=:condition,X=r"channel",
            crossval=:sid, seed=hash((params,seed)))
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

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

# Timeline across salience x target time (for hit trials)
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

# Timeline across salience x target time (for hit trials) with baseline
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

# Individual-data timeline
# -----------------------------------------------------------------

R"""
pl = ggplot(filter($predict,hit == 'hit'),aes(x=winstart,y=correct_mean*100,
        color=interaction(salience_label,target_time_label))) +
    geom_line() + facet_grid(sid~condition) +
    geom_abline(slope=0,intercept=50,linetype=2) +
    guides(fill = guide_legend(title="Salience x Target time"),
            color = guide_legend(title="Salience x Target time")) +
    scale_fill_brewer(palette='Paired') +
    scale_color_brewer(palette='Paired') +
    coord_cartesian(ylim=c(0,100))
# pl
"""

R"""
ggsave(file.path($dir,"object_salience_timeline_ind.pdf"),pl,width=11,height=32)
"""

# Timeline across salience x target time (for miss trials)
# -----------------------------------------------------------------

band = @_ predict |>
    filter(_.hit == "miss",__) |>
    groupby(__,[:winstart,:salience_label,:target_time_label,:condition]) |> #,:before]) |>
    combine(:correct_mean => function(correct)
        bs = bootstrap(mean,correct,BasicSampling(10_000))
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
    geom_abline(slope=0,intercept=50,linetype=2) +
    guides(fill = guide_legend(title="Salience x Target time"),
           color = guide_legend(title="Salience x Target time")) +
    scale_fill_brewer(palette='Paired') +
    scale_color_brewer(palette='Paired') +
    coord_cartesian(ylim=c(40,100))
pl
"""

R"""
ggsave(file.path($dir,"object_salience_timeline_miss.pdf"),pl,width=11,height=8)
"""

# Timeline across salience x target time (for miss trials) with baseline
# -----------------------------------------------------------------

band = @_ predict_baseline |>
    filter(_.hit == "miss",__) |>
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
    scale_fill_brewer(palette='Paired') +
    scale_color_brewer(palette='Paired')
    # coord_cartesian(ylim=c(40,100))
pl
"""

R"""
ggsave(file.path($dir,"object_salience_timeline_miss.pdf"),pl,width=11,height=8)
"""

# Timeline across salineece / target-time - baseline - miss
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
    scale_fill_brewer(palette='Paired') +
    scale_color_brewer(palette='Paired') +
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

# Timeline across salience, using baseline
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

# Timeline across salience - miss - base
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

# Timeline across target time - miss - base
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

# Timeline dividied into data-driven early/late phase
# =================================================================

# select a validation set
late_boundary = 3.5
# valids = StatsBase.sample(MersenneTwister(hash((2019_11_19,:early_boundary))),
#     unique(predict.sid), round(Int,0.3length(unique(predict.sid))), replace=false)
valids = unique(predict.sid)
early_boundary_data = @_ predict |>
    filter(_.sid ∈ valids,__) |> # use a validation set (to avoid "double-dipping" the data)
    filter(_.winstart <= late_boundary,__) |> # aribtrary cutoff, for now...
    groupby(__,[:winstart,:salience_label,:target_time_label,:condition]) |>
    combine(__,:correct_mean => mean => :correct_mean)

# compute a boundary that minimizes within-time-block standard deviation across all groups
boundaries = early_boundary_data.winstart |> unique
border = 2
early_boundary_ind = map(boundaries[(1+border):(end-border)]) do boundary
    df = @_ early_boundary_data |>
        transform!(__,:winstart => (x -> ifelse.(x .< boundary,"early","late")) =>
            :winstart_label) |>
        groupby(__,[:winstart_label,:salience_label,:target_time_label,:condition]) |>
        combine(__,:correct_mean => std => :correct_sd) |>
        mean(__.correct_sd)
end |> argmin
early_boundary = boundaries[border+early_boundary_ind]

# Target-time x salience into early/late windowstart
# -----------------------------------------------------------------

grouped = @_ predict_bounds |>
    filter(_.hit == "hit",__) |>
    groupby(__,[:winstart_label,:target_time_label,:salience_label,:condition,:sid]) |>
    combine(__,:correct_mean => mean => :correct_mean)

R"""
pos = position_dodge(width=0.75)
posjit = position_jitterdodge(jitter.width=0.1,dodge.width=0.75)
pl = ggplot($grouped,
        aes(x    = target_time_label,
            y    = 100*correct_mean,
            fill = interaction(winstart_label, salience_label))) +
    stat_summary(geom='bar',fun.data='mean_cl_boot',
        aes(fill=interaction(winstart_label, salience_label)), width=0.6, position=pos) +
    stat_summary(geom='linerange',fun.data='mean_cl_boot', fun.args=list(conf.int=0.682),
        aes(fill=interaction(winstart_label, salience_label)), width=0.6, position=pos) +
    geom_point(aes(group=interaction(winstart_label, salience_label)), position=posjit,
        alpha=0.4) +
    scale_fill_brewer( palette = 'Paired', direction = -1) +
    scale_color_brewer(palette = 'Paired', direction = -1) +
    guides(fill  = guide_legend(title = "Window time x Salience"),
           color = guide_legend(title = "Window time x Salience")) +
    facet_wrap(~condition) +
    coord_cartesian(ylim=c(0,100))
"""

R"""
model = lm(correct_mean ~ target_time_label*winstart_label*salience_label, $grouped)
anova(model)
summary(model)
"""

R"""
ggsave(file.path($dir,"salience_target_time_bar.pdf"),pl,width=11,height=8)
"""

# Target-time x salience with late - early window difference
# -----------------------------------------------------------------

diffs = @_ predict_bounds |>
    filter(_.hit == "hit",__) |>
    groupby(__,[:winstart_label,:target_time_label,:salience_label,:condition,:sid]) |>
    combine(__,:correct_mean => mean => :correct_mean) |>
    unstack(__,[:target_time_label,:salience_label,:condition,:sid],:winstart_label,:correct_mean) |>
    transform!(__,[:late,:early] => (-) => :diff) |>
    transform!(__,:target_time_label =>
        (t -> recode(t,
            "early" => "2 or fewer switches",
            "late" => "3 or more switches")) => :target_time_descrip)

R"""
pos = position_dodge(width=0.75)
posjit = position_jitterdodge(jitter.width=0.1,dodge.width=0.75)
pl = ggplot($diffs,
        aes(x    = target_time_descrip,
            y    = 100*diff,
            fill = salience_label)) +
    stat_summary(geom='bar',fun.data='mean_cl_boot',
        aes(fill = salience_label), width=0.6, position=pos) +
    stat_summary(geom='linerange',fun.data='mean_cl_boot', fun.args=list(conf.int=0.682),
        aes(fill=salience_label), width=0.6, position=pos) +
    geom_point(aes(group=salience_label), position=posjit,
        alpha=0.4) +
    scale_fill_brewer( palette = 'Set1') +
    scale_color_brewer(palette = 'Set1') +
    guides(fill  = guide_legend(title = "Salience"),
           color = guide_legend(title = "Salience")) +
    facet_wrap(~condition) +
    ylab("Late - Early Window difference (% Correct Classification)") +
    xlab("Target Timing")
"""

R"""
ggsave(file.path($dir,"salience_target_time_diff_bar.pdf"),pl,width=11,height=8)
"""


R"""
pos = position_dodge(width=0.75)
posjit = position_jitterdodge(jitter.width=0.1,dodge.width=0.75)
pl = ggplot($diffs,
        aes(x    = target_time_descrip,
            y    = 100*diff,
            fill = salience_label)) +
    stat_summary(geom='bar',fun.data='mean_cl_boot',
        aes(fill = salience_label), width=0.6, position=pos) +
    stat_summary(geom='linerange',fun.data='mean_cl_boot', fun.args=list(conf.int=0.682),
        aes(fill=salience_label), width=0.6, position=pos) +
    scale_fill_brewer( palette = 'Set1') +
    scale_color_brewer(palette = 'Set1') +
    guides(fill  = guide_legend(title = "Salience"),
           color = guide_legend(title = "Salience")) +
    facet_wrap(~condition) +
    ylab("Late - Early Window difference (% Correct Classification)") +
    xlab("Target Timing")
"""


R"""
ggsave(file.path($dir,"salience_target_time_diff_bar_noind.pdf"),pl,width=11,height=8)
"""

# Salience grouped into early/late windowstart
# -----------------------------------------------------------------
grouped = @_ predict_bounds |>
    # filter(_.winstart <= late_boundary,__) |>
    filter(_.hit == "hit",__) |>
    # filter(_.sid ∉ valids,__) |>
    # transform!(__,:winstart => (x -> ifelse.(x .< early_boundary,"early","late")) =>
    #     :winstart_label) |>
    groupby(__,[:winstart_label,:salience_label,:condition,:sid]) |>
    combine(__,:correct_mean => mean => :correct_mean) |>
    groupby(__,[:winstart_label,:salience_label,:condition]) |>
    combine(:correct_mean => function(correct)
        bs = bootstrap(mean,correct,BasicSampling(10_000))
        μ,low,high = 100 .* confint(bs,BasicConfInt(0.682))[1]
        (correct = μ, low = low, high = high)
    end,__) #|>

R"""
pos = position_dodge(width = 0.75)
pl = ggplot($grouped, aes(x = winstart_label, y = correct, fill = salience_label)) +
    geom_bar(stat = 'identity', aes(fill = salience_label), width = 0.6, position = pos) +
    geom_linerange(aes(ymin = low, ymax = high), position = pos) +
    scale_fill_brewer(name='Salience',palette = 'Set1') +
    ylab('% Correct Classification') +
    xlab('Window Timing') +
    facet_wrap(~condition) + coord_cartesian(ylim = c(50, 100))
"""

R"""
ggsave(file.path($dir,"salience_bar.pdf"),pl,width=11,height=8)
"""

# Target-timing grouped into early/late windowstart
# -----------------------------------------------------------------

grouped = @_ predict_bounds |>
    filter(_.hit == "hit",__) |>
    # filter(_.sid ∉ valids, __) |>
    # filter(_.winstart <= late_boundary, __) |>
    # transform!(__, :winstart => (x -> ifelse.(x .< early_boundary, "early", "late"))
    #     => :winstart_label) |>
    groupby(__, [:winstart_label, :target_time_label, :condition, :sid]) |>
    combine(__, :correct_mean => mean => :correct_mean) |>
    groupby(__, [:winstart_label, :target_time_label, :condition]) |>
    combine(:correct_mean => function(correct)
        bs = bootstrap(mean,correct, BasicSampling(10_000))
        μ, low, high = 100 .* confint(bs,BasicConfInt(0.682))[1]
        (correct = μ, low = low, high = high)
    end,__) #|>


R"""
pos = position_dodge(width = 0.75)
pl = ggplot($grouped,aes(x = winstart_label, y = correct, fill = target_time_label)) +
    geom_bar(stat = 'identity', aes(fill = target_time_label), width = 0.6, position = pos) +
    geom_linerange(aes(ymin = low, ymax = high), position = pos) +
    scale_fill_brewer(
        name = 'Target Timing',
        label=c('2 or fewer switches','3 or more switches'),
        palette='Set2') +
    xlab('Window Timing') +
    ylab('% Correct Classification') +
    facet_wrap(~condition) + coord_cartesian(ylim = c(50, 100))
"""

R"""
ggsave(file.path($dir, "targettime_bar.pdf"), pl, width = 11, height = 8)
"""

# Overall vs Miss
# =================================================================

classdf_file = joinpath(cache_dir(),"data","freqmeans_miss_baseline.csv")
if use_cache && isfile(classdf_file)
    classdf_missbase = CSV.read(classdf_file)
else
    eeg_files = dfhit = @_ readdir(processed_datadir("eeg")) |> filter(occursin(r".mcca$",_), __)
    subjects = Dict(file => load_subject(joinpath(processed_datadir("eeg"), file), stim_info,
                                            encoding = RawEncoding())
        for file in eeg_files)

    classdf_missbase = find_powerdiff(
        subjects,groups=[:salience],
        hittypes = ["miss","hit", "baseline"],
        regions = ["target", "baseline"],
        windows = [(len=len,start=start,before=-len)
            for start in range(0,4,length=64),
                len in best_windows_sal.winlen |> unique])

    CSV.write(classdf_file,classdf)
end

objectdf = @_ classdf_missbase |>
    filter(_.condition in ["global","object"],__) |>
    filter(_1.winlen == winlens[(condition = "object", salience = _1.salience)].winlen[1],__)
paramfile = joinpath(datadir(),"svm_params","object_salience.csv")
best_params = CSV.read(paramfile)
rename!(best_params,:subjects => :sid)
object_predict = classpredict(objectdf, best_params, "object", :salience, :hit)

spatialdf = @_ classdf_missbase |>
    filter(_.condition in ["global","spatial"],__) |>
    filter(_1.winlen == winlens[(condition = "spatial", salience = _1.salience)].winlen[1],__)
paramfile = joinpath(datadir(),"svm_params","spatial_salience.csv")
best_params = CSV.read(paramfile)
rename!(best_params,:subjects => :sid)
spatial_predict = classpredict(spatialdf, best_params, "spatial", :salience, :hit)

function bestonly(var,measure,df)
    means = combine(groupby(df,var),measure => mean => measure)
    bestvar = means[var][argmax(means[measure])]

    @_ filter(_[var] == bestvar,df)
end

hit_compare = @_ vcat(object_predict,spatial_predict) |>
    groupby(__, [:salience, :hit, :condition]) |>
    combine(bestonly(:winstart, :correct_mean, _), __) |>
    groupby(__, [:salience, :condition, :sid, :hit]) |>
    combine(__, :correct_mean => mean => :correct_mean) |>
    groupby(__, :hit) |>
    combine(:correct_mean => function(x)
        bs = bootstrap(mean, x, BasicSampling(10_000))
        μ, low, high = 100 .* confint(bs, BasicConfInt(0.682))[1]
        (correct = μ, low = low, high = high)
    end, __)

R"""
pl = ggplot($hit_compare,aes(
        x = factor(hit,order=T,levels=c('hit','miss','baseline')),
        y = correct,fill=hit)) +
    geom_bar(stat='identity',width=0.7) +
    geom_linerange(aes(ymin=low,ymax=high)) +
    coord_cartesian(ylim=c(40,100)) +
    geom_abline(intercept=50,slope=0,linetype=2) +
    guides(color=F,fill=F) +
    ylab('% Correct') + xlab('')
pl
"""

R"""
ggsave(file.path($dir,"hit_v_miss_v_baseline_beststart.pdf"),pl,width=11,height=8)
"""

hit_compare2 = @_ vcat(object_predict,spatial_predict) |>
    groupby(__,[:salience,:condition,:sid,:hit]) |>
    combine(__,:correct_mean => mean => :correct_mean) |>
    groupby(__,:hit) |>
    combine(:correct_mean => function(x)
        bs = bootstrap(mean,x,BasicSampling(10_000))
        μ,low,high = 100 .* confint(bs,BasicConfInt(0.682))[1]
        (correct = μ, low = low, high = high)
    end,__)

R"""
pl = ggplot($hit_compare2,aes(x=factor(hit,order=T,levels=c('hit','miss','baseline')),
        y=correct,fill=hit)) +
    geom_bar(stat='identity',width=0.7) +
    geom_linerange(aes(ymin=low,ymax=high)) +
    coord_cartesian(ylim=c(40,100)) +
    geom_abline(intercept=50,slope=0,linetype=2) +
    guides(color=F,fill=F) +
    ylab('% Correct') + xlab('')
pl
"""

R"""
ggsave(file.path($dir,"hit_v_miss_v_baseline_grouopavg.pdf"),pl,width=11,height=8)
"""

# Target Timing Timeline
# =================================================================

winlens = groupby(best_windows_tim,[:condition,:target_time])

objectdf = @_ classdf_tim |>
    filter(_.condition in ["global","object"],__) |>
    filter(_1.winlen in spread(winlens[(condition = "object", target_time = _1.target_time)].winlen[1],0.5,6),__)
paramfile = joinpath(datadir(),"svm_params","object_target_time.csv")
best_params = CSV.read(paramfile)
rename!(best_params,:subjects => :sid)
object_predict = classpredict(objectdf, best_params, "object", :target_time)

spatialdf = @_ classdf_tim |>
    filter(_.condition in ["global","spatial"],__) |>
    filter(_1.winlen in spread(winlens[(condition = "spatial", target_time = _1.target_time)].winlen[1],0.5,6),__)
paramfile = joinpath(datadir(),"svm_params","spatial_target_time.csv")
best_params = CSV.read(paramfile)
rename!(best_params,:subjects => :sid)
spatial_predict = classpredict(spatialdf, best_params, "spatial", :target_time)

predict = vcat(object_predict,spatial_predict)

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

band = @_ predict |>
    # filter(_.before == "zero",__) |>
    groupby(__,[:winstart,:target_time,:condition]) |> #,:before]) |>
    combine(:correct_mean => function(correct)
        bs = bootstrap(mean,correct,BasicSampling(10_000))
        μ,low,high = 100 .* confint(bs,BasicConfInt(0.682))[1]
        (correct = μ, low = low, high = high)
    end,__) #|>
    # transform!(__,[:target_time,:before] =>
    #     ((x,y) -> string.(x,"_",y)) => :target_time_for)

R"""
pl = ggplot($band,aes(x=winstart,y=correct,color=target_time)) +
    geom_ribbon(aes(ymin=low,ymax=high,fill=target_time,color=NULL),alpha=0.4) +
    geom_line() + facet_grid(~condition) +
    geom_abline(slope=0,intercept=50,linetype=2) +
    coord_cartesian(ylim=c(40,100))
"""

R"""
ggsave(file.path($dir,"object_target_time_timeline.pdf"),pl,width=11,height=8)
"""

best_wins = @_ predict |>
    groupby(__,[:winstart,:target_time,:condition]) |>
    combine(__,:correct_mean => mean => :correct) |>
    groupby(__,[:target_time,:condition]) |>
    combine(__,[:winstart,:correct] =>
        ((win,val) -> win[argmax(val)]) => :winstart) |>
    groupby(__,:condition)

subj_sum = @_ predict |>
    filter(_.winstart in best_wins[(condition = _.condition,)].winstart, __) |>
    groupby(__,[:winstart,:target_time,:condition]) |>
    combine(:correct_mean => function(correct)
        bs = bootstrap(mean,correct,BasicSampling(10_000))
        μ,low,high = 100 .* confint(bs,BasicConfInt(0.682))[1]
        (correct = μ, low = low, high = high)
    end,__)

subj_sum = @_ subj_sum |>
    groupby(__,[:condition,:winstart]) |>
    transform!(__,[:condition,:winstart] => function(cond,start)
        best = best_wins[(condition = cond[1],)]
        bestfor = filter(x -> x.winstart == start[1],best)
        fill(bestfor.target_time[1], length(start))
    end => :bestfor)

subj_sum = @_ subj_sum |>
    groupby(__,[:condition,:winstart,:bestfor]) |>
    transform!(__,[:winstart,:bestfor] => function(start,bestfor)
        @show start
        label = "Best Time \nfor $(uppercasefirst(bestfor[1]))\n"*
            "($(round(start[1],digits=2)) s)"
        fill(label,length(start))
    end => :label)

R"""
pl = plot_grid(
    ggplot(filter($subj_sum,condition == "object"),
        aes(x=label,y=correct,color=target_time,group=target_time)) +
        geom_bar(stat='identity',position=position_dodge(width=0.8),aes(fill=target_time),width=0.6) +
        geom_linerange(aes(ymin=low,ymax=high),position=position_dodge(width=0.8),color='black') +
        guides(color=F,fill=F) + coord_cartesian(ylim=c(40,100)) +
        geom_abline(intercept = 50, slope = 0, linetype = 2) +
        xlab("") + ylab("% Correct"),
    ggplot(filter($subj_sum,condition == "spatial"),
        aes(x=label,y=correct,color=target_time,group=target_time)) +
        geom_bar(stat='identity',position=position_dodge(width=0.8),aes(fill=target_time),width=0.6) +
        geom_linerange(aes(ymin=low,ymax=high),position=position_dodge(width=0.8),color='black') +
        coord_cartesian(ylim=c(40,100)) +
        geom_abline(intercept = 50, slope = 0, linetype = 2) +
        xlab("") + ylab("% Correct"),
        rel_widths = c(1,1.2),
    labels = c('Object', 'Spatial'), label_x=0.5, hjust=0.5)
"""

R"""
ggsave(file.path($dir,"target_time_classify_summary.pdf"),pl,width=8,height=3)
"""
