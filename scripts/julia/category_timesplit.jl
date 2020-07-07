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
R"library(ggplot2)"
R"library(cowplot)"
R"library(lsr)"

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

paramdir = processed_datadir("svm_params")
classfile = joinpath(paramdir, savename("timeline-classify",
    (absolute = use_absolute_features,), "json"))

predict = CSV.read(classfile)

# Time split selection
# =================================================================

# select a validation set
validation_ids = StatsBase.sample(MersenneTwister(hash((seed, :early_boundary))),
    unique(predict.sid), round(Int, 0.2length(unique(predict.sid))), replace = false)
# validation_ids = unique(predict.sid)
boundary_selection_data = @_ predict |>
    filter(_.winstart > 0 && _.winstart < 2.8, __) |>
    # use a validation set (to avoid "double-dipping" the data)
    filter(_.sid ∈ validation_ids, __) |>
    groupby(__, [:winstart]) |>
    combine(__, :correct_mean => mean => :correct_mean) |>
    sort!(__, :winstart)

lowpass = digitalfilter(Lowpass(0.1), Butterworth(5))
boundary_selection_data[!, :correct_mean_lp] =
    filtfilt(lowpass, boundary_selection_data.correct_mean)

split_time = @_ boundary_selection_data |>
    sort!(__, :winstart) |>
    combine(__, [:correct_mean_lp, :winstart] =>
        ((x, t) -> t[argmax(diff(x))+1]) => :maxtime) |>
    only(__.maxtime)

before_time = @_ boundary_selection_data |>
    filter(_.winstart < split_time, __) |>
    combine(__, [:correct_mean_lp, :winstart] =>
        ((x, t) -> t[argmax(x)]) => :before) |>
    only(__.before)

after_time = @_ boundary_selection_data |>
    filter(_.winstart > split_time, __) |>
    combine(__, [:correct_mean_lp, :winstart] =>
        ((x, t) -> t[argmax(x)]) => :after) |>
    only(__.after)


after_time = @_ boundary_selection_data.winstart |> unique |> sort! |> __[25]

R"""
ggplot($boundary_selection_data, aes(x = winstart, y = correct_mean)) + geom_line() +
    geom_line(aes(y = correct_mean_lp), alpha = 0.5) +
    geom_vline(xintercept = $split_time,  linetype = 2, color = 'red') +
    geom_vline(xintercept = $before_time, linetype = 2, color = 'gray') +
    geom_vline(xintercept = $after_time,  linetype = 2, color = 'gray')
"""

R"""
ggsave(file.path($dir,"window_time_selection.pdf"))
"""

# Plots
# =================================================================

# Salience x window time
# -----------------------------------------------------------------

# find best performance for each condition before and after the selected split
salience_df = @_ predict |>
    filter(_.sid ∉ validation_ids,__) |>
    filter(_.winstart ∈ [before_time, after_time], __) |>
    transform!(__, :winstart => (x -> ifelse.(x .<= split_time, "early", "late")) =>
        :winstart_label) |>
    groupby(__, [:winstart_label, :condition, :salience_label, :sid]) |>
    combine(__, :correct_mean => mean => :correct_mean)

R"""
pos = position_dodge(width = 0.6)
pl = ggplot($salience_df, aes(x = winstart_label, y = correct_mean, group = salience_label)) +
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
ggsave(file.path($dir, "salience_bar.pdf"), pl, width = 8, height = 6)
"""

R"""
model = lm(correct_mean ~ salience_label * winstart_label * condition,$salience_df)
print(summary(model))
print(anova(model))
print(etaSquared(model))
"""

# Target timing
# -----------------------------------------------------------------

# find best performance for each condition before and after the selected split
target_time_df = @_ predict |>
    filter(_.sid ∉ validation_ids,__) |>
    filter(_.winstart ∈ [before_time, after_time], __) |>
    transform!(__, :winstart => (x -> ifelse.(x .<= split_time, "early", "late")) =>
        :winstart_label) |>
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

R"""
model = lm(correct_mean ~ target_time_label * condition,$target_time_df)
print(summary(model))
print(anova(model))
print(etaSquared(model))
"""

# Salience x target timing
# -----------------------------------------------------------------

salience_target_df = @_ predict |>
    filter(_.sid ∉ validation_ids,__) |>
    filter(_.winstart ∈ [before_time, after_time], __) |>
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

R"""
model = lm(correct_mean ~ target_time_label * salience_label * condition,$salience_target_df)
print(summary(model))
print(anova(model))
print(etaSquared(model))
"""
