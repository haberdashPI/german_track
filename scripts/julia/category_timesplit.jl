# Setup
# =================================================================

using DrWatson
@quickactivate("german_track")
seed = 072189
use_absolute_features = true

using GermanTrack
using EEGCoding
using RCall
using DataFrames
using Underscores
using Alert
using DSP
using StatsBase
using Random
using Dates
using Peaks

R"library(ggplot2)"
R"library(cowplot)"
R"library(lsr)"
R"library(multcomp)"
R"library(rstanarm)"
R"library(Hmisc)"

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

paramdir = processed_datadir("svm_params")
classfile = joinpath(paramdir, savename("timeline-classify",
    (absolute = use_absolute_features,), "csv"))

predictdf = CSV.read(classfile)

# Time split selection
# =================================================================

# select a validation set
# we use a validation set (to avoid "double-dipping" the data)

validation_ids = StatsBase.sample(MersenneTwister(hash((seed, :early_boundary))),
    unique(predictdf.sid), round(Int, 0.2length(unique(predictdf.sid))), replace = false)
# validation_ids = unique(predictdf.sid)
lowpass = digitalfilter(Lowpass(0.45), Butterworth(5))
boundary_selection_data = @_ predictdf |>
    filter(_.winstart >= 0 && _.winstart < 2.0,__) |>
    filter(_.sid ∈ validation_ids, __) |>
    filter(_.hit == "hit", __) |>
    groupby(__, [:winstart,:condition]) |>
    combine(__, :correct_mean => mean => :correct_mean) |>
    sort!(__, :winstart) |>
    groupby(__,[:condition]) |>
    transform!(__,:correct_mean => (x -> filtfilt(lowpass, x)) => :correct_mean_lp)

split_times = @_ boundary_selection_data |>
    groupby(__,[:condition]) |>
    combine(__, [:correct_mean_lp, :winstart] =>
        (function (x, t)
            dx = abs.(diff(x))
            max = maxima(dx)
            t[max[argmax(dx[max])]]
        end) => :pos)
splitg = groupby(split_times,:condition)

function mymin(x, t)
    m = minima(x)
    isempty(m) ? t[argmin(x)] : t[m[end]]
end
before_time = @_ boundary_selection_data |>
    filter(_.winstart < splitg[(condition = _.condition,)].pos[1], __) |>
    groupby(__,[:condition]) |>
    combine(__, [:correct_mean_lp, :winstart] =>
        ((x, t) -> t[maxima(x)[end]]) => :pos)

after_time = @_ boundary_selection_data |>
    filter(_.winstart >= splitg[(condition = _.condition,)].pos[1], __) |>
    groupby(__,[:condition]) |>
    combine(__, [:correct_mean_lp, :winstart] =>
        ((x, t) -> t[maxima(x)[1]]) => :pos)

R"""
pl1 = ggplot($boundary_selection_data, aes(x = winstart, y = correct_mean)) + geom_line() +
    geom_line(aes(y = correct_mean_lp), alpha = 0.5) +
    geom_vline(data = $split_times, aes(xintercept = pos), linetype = 2, color = 'red') +
    geom_vline(data = $before_time, aes(xintercept = pos), linetype = 2, color = 'gray') +
    geom_vline(data = $after_time,  aes(xintercept = pos), linetype = 2, color = 'gray') +
    facet_wrap(~condition)
"""

cont_salience_df = @_ predictdf |>
    filter(_.sid ∉ validation_ids, __) |>
    filter(_.hit == "hit",__) |>
    filter(_.winstart > 0,__) |>
    groupby(__, [:winstart, :condition, :salience_label, :sid]) |>
    combine(__, :correct_mean => mean => :correct_mean)

R"""
pl2 = ggplot($cont_salience_df, aes(x = winstart, y = correct_mean, group = salience_label)) +
    stat_summary(geom='line', aes(color = salience_label)) +
    stat_summary(geom='ribbon', alpha = 0.4, aes(fill = salience_label)) +
    facet_grid(~condition) +
    geom_vline(data = $split_times, aes(xintercept = pos), linetype = 2, color = 'red') +
    geom_vline(data = $before_time, aes(xintercept = pos), linetype = 2, color = 'gray') +
    geom_vline(data = $after_time,  aes(xintercept = pos), linetype = 2, color = 'gray') +
    coord_cartesian(xlim=c(0,2))
"""

R"""
pl = plot_grid(pl1, pl2, nrow=2, axis = "lr", align = "hv")
ggsave2(file.path($dir, "window_time_selection.pdf"), pl)
"""


# Plots
# =================================================================

# Salience x window time
# -----------------------------------------------------------------

# find best performance for each condition before and after the selected split
beforeg = groupby(before_time, :condition)
afterg  = groupby(after_time,  :condition)
splitg = groupby(split_times, :condition)

salience_df = @_ predictdf |>
    filter(_.sid ∉ validation_ids,__) |>
    filter(_.hit == "hit",__) |>
    filter(_.winstart >= 0,__) |>
    transform!(__, [:winstart, :condition] =>
        ByRow((x, c) -> x == only(beforeg[(condition = c,)].pos) ? "early" :
                        x == only(afterg[( condition = c,)].pos) ? "late"  : missing) =>
            :winstart_label) |>
    # transform!(__, [:winstart, :condition] =>
    #     ByRow((t, c) -> (-0.75 < (splitg[(condition = c,)].pos[1] - t) < -0.2) ? "late" :
    #                     (0.2 < (splitg[( condition = c,)].pos[1] - t) < 0.75) ? "early" :
    #                     missing) => :winstart_label) |>
    filter(!ismissing(_.winstart_label), __) |>
    groupby(__, [:winstart_label, :condition, :salience_label, :sid]) |>
    combine(__, :correct_mean => mean => :correct_mean)

R"""
pos = position_dodge(width = 0.6)
pl = ggplot($salience_df, aes(x = winstart_label, y = correct_mean, group = salience_label)) +
    stat_summary(fun.data = 'mean_se', aes(fill = salience_label), geom = 'bar',
        position = pos, width = 0.5) +
    stat_summary(fun.data = 'mean_se', aes(fill = salience_label), geom = 'linerange',
        # fun.args = list(conf.int = 0.95),
        position = pos) +
    geom_point(alpha = 0.4, aes(fill = salience_label),
        position = position_jitterdodge(jitter.width = 0.05, dodge.width = 0.6)) +
    facet_wrap(~condition) +
    geom_hline(yintercept = 0.5, linetype = 2) +
    scale_fill_brewer(palette = 'Set1') +
    coord_cartesian(ylim = c(0.3, 1))
"""

R"""
ggsave(file.path($dir, "salience_bar.pdf"), pl, width = 8, height = 6)
"""

CSV.write(joinpath(processed_datadir("analyses"), "spatial-timing.csv"), salience_df)
objdf = @_ filter(_.condition == "object", salience_df)
R"""
library()
df = $objdf
df$correct_mean = (df$correct_mean - 0.5)*0.99 + 0.5
model = glm(correct_mean ~ salience_label * winstart_label, df, family = mgcv::betar)
print(summary(model))
print(anova(model))
print(summary(aov(correct_mean ~ salience_label * winstart_label + Error(sid / (salience_label/winstart_label)), $objdf)))
print(etaSquared(model))
"""

spadf = @_ filter(_.condition == "spatial", salience_df)
R"""
model = lm(correct_mean ~ salience_label * winstart_label,$spadf)
print(summary(model))
print(anova(model))
print(etaSquared(model))
print(summary(aov(correct_mean ~ salience_label * winstart_label + Error(sid / (salience_label/winstart_label)), $spadf)))
"""

R"""
pos = position_dodge(width = 0.6)
pl = ggplot($salience_df, aes(x = winstart_label, y = correct_mean, group = salience_label)) +
    stat_summary(fun.data = 'mean_se', aes(fill = salience_label), geom = 'bar',
        position = pos, width = 0.5) +
    stat_summary(fun.data = 'mean_se', aes(fill = salience_label), geom = 'linerange',
        # fun.args = list(conf.int = 0.95),
        position = pos) +
    geom_point(alpha = 0.4, aes(fill = salience_label),
        position = position_jitterdodge(jitter.width = 0.05, dodge.width = 0.6)) +
    facet_wrap(~condition) +
    geom_hline(yintercept = 0.5, linetype = 2) +
    scale_fill_brewer(palette = 'Set1') +
    coord_cartesian(ylim = c(0.3, 1))
"""

# Target salience x target time
# -----------------------------------------------------------------

salience_target_time_df = @_ predictdf |>
    filter(_.sid ∉ validation_ids,__) |>
    filter(_.hit == "hit",__) |>
    filter(_.winstart >= 0,__) |>
    transform!(__, [:winstart, :condition] =>
        ByRow((x, c) -> x == only(beforeg[(condition = c,)].pos) ? "early" :
                        x == only(afterg[( condition = c,)].pos) ? "late"  : missing) =>
            :winstart_label) |>
    # transform!(__, [:winstart, :condition] =>
    #     ByRow((t, c) -> (-0.75 < (splitg[(condition = c,)].pos[1] - t) < -0.2) ? "late" :
    #                     (0.2 < (splitg[( condition = c,)].pos[1] - t) < 0.75) ? "early" :
    #                     missing) => :winstart_label) |>
    filter(!ismissing(_.winstart_label), __) |>
    groupby(__, [:winstart_label, :target_timeLabel, :condition, :salience_label, :sid]) |>
    combine(__, :correct_mean => mean => :correct_mean)

# Target timing
# -----------------------------------------------------------------

# todo: how do we pick the right times here? it should be something different
# just do 0?

# find best performance for each condition before and after the selected split
target_time_df = @_ predictdf |>
    filter(_.winstart == 0,__) |>
    groupby(__, [:condition, :target_time_label, :sid]) |>
    combine(__, :correct_mean => mean => :correct_mean)

R"""
pos = position_dodge(width = 0.6)
pl = ggplot($target_time_df, aes(
        x = target_time_label,
        y = correct_mean,
        group = target_time_label)) +
    stat_summary(fun.data = 'mean_cl_boot', aes(fill = target_time_label),
        geom = 'bar', position = pos, width = 0.5) +
    stat_summary(fun.data = 'mean_cl_boot', aes(fill = target_time_label),
        geom = 'linerange', fun.args = list(conf.int = 0.682), position = pos) +
    geom_point(alpha = 0.4, aes(fill = target_time_label),
        position = position_jitterdodge(jitter.width = 0.05, dodge.width = 0.6)) +
    facet_wrap(~condition) +
    geom_hline(yintercept = 0.5, linetype = 2) +
    scale_fill_brewer(palette = 'Set2') +
    coord_cartesian(ylim = c(0.3, 1))
"""

R"""
ggsave(file.path($dir, "target_time_bar.pdf"), pl, width = 8, height = 6)
"""

CSV.write(joinpath(processed_datadir("analyses"), "target-time.csv"), target_time_df)
R"""
model = lm(correct_mean ~ target_time_label * condition,$target_time_df)
print(summary(model))
print(anova(model))
print(etaSquared(model))
print(summary(aov(correct_mean ~ target_time_label +
    Error(sid / target_time_label), $target_time_df)))
"""

# Salience x target timing
# -----------------------------------------------------------------

salience_target_df = @_ predictdf |>
    filter(_.winstart == 0,__) |>
    groupby(__, [:condition, :target_time_label, :salience_label, :sid]) |>
    combine(__, :correct_mean => mean => :correct_mean)

R"""
pos = position_dodge(width = 0.6)
pl = ggplot($salience_target_df, aes(
        x = target_time_label,
        y = correct_mean,
        group = salience_label)) +
    stat_summary(fun.data = 'mean_cl_boot', aes(fill = salience_label), geom = 'bar',
        position = pos, width = 0.5) +
    stat_summary(fun.data = 'mean_cl_boot', aes(fill = salience_label), geom = 'linerange',
        fun.args = list(conf.int = 0.682), position = pos) +
    geom_point(alpha = 0.4, aes(fill = salience_label),
        position = position_jitterdodge(jitter.width = 0.05, dodge.width = 0.6)) +
    facet_wrap(~condition) +
    geom_hline(yintercept = 0.5, linetype = 2) +
    scale_fill_brewer(palette = 'Set1') +
    coord_cartesian(ylim = c(0.3, 1))
"""

R"""
ggsave(file.path($dir, "salience_target_time_bar.pdf"), pl, width = 8, height = 6)
"""

CSV.write(joinpath(processed_datadir("analyses"), "salience-target-time.csv"),
    salience_target_df)

R"""
model = lm(correct_mean ~ target_time_label * salience_label * condition,$salience_target_df)
print(summary(model))
print(anova(model))
print(etaSquared(model))
print(summary(aov(correct_mean ~ salience_label * target_time_label +
    Error(sid / (salience_label/target_time_label)), $salience_target_df)))
"""
