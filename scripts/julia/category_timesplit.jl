# Setup
# =================================================================

using DrWatson
@quickactivate("german_track")
seed = 072189
use_absolute_features = true
classifier = :logistic_l1

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

paramdir = processed_datadir("classifier_params")
classfile = joinpath(paramdir, savename("timeline-classify",
    (absolute = use_absolute_features, classifier = classifier), "csv"))

predictdf = CSV.read(classfile)

# Plots
# =================================================================

# Window start selection
# -----------------------------------------------------------------

validation_ids = StatsBase.sample(MersenneTwister(stablehash(seed, :winstart)),
    unique(predictdf.sid), round(Int, 0.1length(unique(predictdf.sid))), replace = false)
# validation_ids = unique(predictdf.sid)
lowpass = digitalfilter(Lowpass(0.5), Butterworth(5))
boundary_selection_data = @_ predictdf |>
    filter(_.winstart > 0.2 && _.winstart < 2.0,__) |>
    filter(_.sid ∈ validation_ids, __) |>
    filter(_.hit == "hit", __) |>
    groupby(__, [:winstart]) |>
    combine(__, :correct_mean => mean => :correct_mean) |>
    sort!(__, :winstart) |>
    transform!(__,:correct_mean => (x -> filtfilt(lowpass, x)) => :correct_mean_lp)

start_times = @_ boundary_selection_data |>
    combine(__, [:correct_mean_lp, :winstart] =>
        ((x, t) -> t[maxima(x)[1]]) => :pos)

R"""
pl1 = ggplot($boundary_selection_data, aes(x = winstart, y = correct_mean)) + geom_line() +
    geom_line(aes(y = correct_mean_lp), alpha = 0.5) +
    geom_vline(data = $start_times, aes(xintercept = pos), linetype = 2, color = 'red')
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
    geom_vline(data = $start_times, aes(xintercept = pos), linetype = 2, color = 'red') +
    coord_cartesian(xlim=c(0,2))
"""

R"""
pl = plot_grid(pl1, pl2, nrow=2, axis = "lr", align = "hv")
ggsave2(file.path($dir, $("window_time_selection_$(classifier).pdf")), pl)
"""

# Target timing
# -----------------------------------------------------------------

# todo: how do we pick the right times here? it should be something different
# just do 0?

# find best performance for each condition before and after the selected split
target_time_df = @_ predictdf |>
    filter(_.winstart == only(start_times.pos),__) |>
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
ggsave(file.path($dir, $("target_time_bar_$(classifier).pdf")), pl, width = 8, height = 6)
"""

# Salience x target timing
# -----------------------------------------------------------------

salience_target_df = @_ predictdf |>
    filter(_.winstart == only(start_times.pos),__) |>
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
ggsave(file.path($dir, $("salience_target_time_bar_$(classifier).pdf")), pl, width = 8, height = 6)
"""

CSV.write(joinpath(processed_datadir("analyses"),
            savename("salience-target-time", (classifier = classifier,), "csv")),
    salience_target_df)
