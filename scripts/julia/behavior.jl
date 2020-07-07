using DrWatson
@quickactivate("german_track")

using GermanTrack
import GermanTrack: stim_info
using VegaLite
using DataFrames
using JSON3
using BangBang
using Underscores
using Dates
using Statistics
using Distributions
using RCall
R"library(ggplot2)"

df = @_ filter(occursin(r"\.h5$", _), readdir(processed_datadir("eeg"))) |>
        mapreduce(events_for_eeg(_, stim_info), append!!, __) |>
        insertcols!(__,:hit => ishit.(eachrow(__),region = "target")) |>
        transform!(__,:hit => (x -> in.(x,Ref(["hit", "reject"]))) => :correct) |>
        categorical!

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

meanb(x,n=1) = (sum(x)+(n/2))/(length(x)+n)
function dprime(hits,falarm,n=1)
    quantile(Normal(),meanb(hits,n)) - quantile(Normal(),meanb(falarm,n))
end

dfsum = @_ df |>
    groupby(__,[:sid,:condition]) |>
    combine(__,
        :correct                    => mean,
        [:target_present, :correct] => ((t,cor) -> mean(  t .&   cor)) => :truepos,
        [:target_present, :correct] => ((t,cor) -> mean(.!t .&   cor)) => :trueneg,
        [:target_present, :correct] => ((t,cor) -> mean(.!t .& .!cor)) => :falsepos,
        [:target_present, :correct] => ((t,cor) -> mean(  t .& .!cor)) => :falseneg,
        [:target_present, :correct] => ((t,cor) -> dprime(t .& cor, t .& .!cor)) => :dp)

R"""
pl = ggplot($dfsum, aes(x = condition, y = dp, group = condition)) +
    stat_summary(fun.data = 'mean_cl_boot', geom = 'pointrange',
        fun.args = list(conf.int = 0.682), size = 1, aes(color = condition)) +
    geom_point(size = 1, alpha = 0.4, position = position_jitter(width = 0.1))
"""

R"""
ggsave(file.path($dir, "behavior_summary.pdf"), pl, width = 8, height = 6)
"""

condition_bytrues = dfsum |>
    @vlplot(
        mark={:point,filled=true}, column=:condition,
        x=:falsepos, y=:truepos)
save(joinpath(dir,"behavior_summary_splitcor.pdf"),condition_bytrues)

condition_bytrues_sid = dfsum |>
    @vlplot(
        mark={:text}, column=:condition,
        text=:sid,
        x=:falsepos, y=:truepos)

condition_byfalse = dfsum |>
    @vlplot(
        mark={:point,filled=true}, column=:condition,
        x=:falsepos, y=:dp)

dftiming = @_ df |>
    transform!(__, :target_time => (x -> 1.2floor.(Int,x/1.2)) => :time_bin) |>
    groupby(__, [:sid, :condition, :time_bin]) |>
    combine(__, [:target_present, :correct] =>
        ((t,cor) -> dprime(t .& cor, t .& .!cor)) => :dp)
timing = dftiming |>
    @vlplot(:line,x=:time_bin,y=:dp,color="sid:o",column=:condition)
save(joinpath(dir,"behavior_bytype.pdf"),timing)

df_salience_time = @_ df |>
    filter(_.target_present,__) |>
    groupby(__, [:sid, :condition, :salience_label, :target_time_label]) |>
    combine(__, :correct => mean)

R"""
pl = ggplot($df_salience_time, aes(
        x     = target_time_label,
        y     = correct_mean,
        group = salience_label,
        color = salience_label
    )) +
    stat_summary(fun.data = 'mean_cl_boot', geom = 'pointrange',
        fun.args = list(conf.int = 0.682), size = 1,
        position = position_dodge(width = 0.3)) +
    geom_point(size = 1, alpha = 0.4,
        position = position_jitterdodge(dodge.width = 0.5, jitter.width = 0.1)) +
    scale_color_brewer(palette='Set1') +
    facet_wrap(~condition)
"""

R"""
ggsave(file.path($dir, "behavior_salience_timing.pdf"), pl, width = 8, height = 6)
"""
