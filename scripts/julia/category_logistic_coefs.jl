using DrWatson
@quickactivate("german_track")
seed = 072189
use_absolute_features = true
classifier = :logistic_l1
n_winlens = 6

using EEGCoding,
    GermanTrack,
    DataFrames,
    Statistics,
    Dates,
    Underscores,
    Random,
    Printf,
    ProgressMeter,
    FileIO,
    StatsBase,
    RCall,
    BangBang,
    Transducers,
    PyCall,
    Alert,
    JSON3,
    JSONTables,
    Formatting,
    ScikitLearn

import GermanTrack: stim_info, speakers, directions, target_times, switch_times

wmeanish(x,w) = iszero(sum(w)) ? 0.0 : mean(coalesce.(x,one(eltype(x))/2),weights(w))

R"""
library(ggplot2)
library(cowplot)
library(dplyr)
library(Hmisc)
"""

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

paramdir = processed_datadir("classifier_params")
best_windows_file = joinpath(paramdir,savename("best-windows",
    (absolute = use_absolute_features, classifier = classifier), "json"))
best_windows = jsontable(open(JSON3.read,best_windows_file,"r")[:data]) |> DataFrame

paramdir    = processed_datadir("classifier_params")
paramfile   = joinpath(paramdir,savename("hyper-parameters",
    (absolute = use_absolute_features, classifier = classifier),"json"))

params = classifier_param_names(classifier)
best_params = @_ jsontable(open(JSON3.read, paramfile, "r")[:data]) |> DataFrame |>
    groupby(__, :fold) |>
    combine(__, (params .=> first .=> params)...) |>
    combine(__, (params .=> mean  .=> params)...)
# select_params = best_params[1,:]
select_params = (C = 0.05, )

isdir(processed_datadir("features")) || mkdir(processed_datadir("features"))
classdf_file = joinpath(processed_datadir("features"), savename("freaqmeans",
    (absolute = use_absolute_features,), "csv"))

classdf_file = joinpath(cache_dir(),"data",
    savename("freqmeans_timeline",
        (absolute    = use_absolute_features,
         classifier  = classifier,
         n_winlens   = n_winlens,
         winstart_max  = 2,
         n_winstarts = 24),
        "csv"))

classdf = @_ CSV.File(classdf_file) |> DataFrame! |> filter(_.winstart == 0, __)

coef_names = names(classdf[:,r"channel"])
function parsecoef(coef)
    parsed = match(r"channel_([0-9]+)_([a-z]+)",coef_names[coef])
    chanstr, freqbin = parsed[1], parsed[2]
    chan = parse(Int,chanstr)
    chan, freqbin
end

# TODO: z-score by freqbin

for freqbin in unique(getindex.(parsecoef.(1:length(coef_names)),2))
    classdf[:,Regex(freqbin)] .= zscore(Array(classdf[:,Regex(freqbin)]))
end

function findcoefs(sdf)
    model, result = runclassifier(
        buildmodel(select_params, classifier, 2017_09_16),
        data = sdf, X = r"channel", y = :condition, seed = seed
    )
    DataFrame(
        correct = wmeanish(result.correct, result.weight),
        bias = model.intercept_;
        (Symbol("coef$(fmt("03d",i))") => c for (i,c) in enumerate(model.coef_))...
    )
end

coefs = vcat(
    @_( classdf |> filter(_.condition in ["global","object"],__) |>
        groupby(__, [:hit, :salience_label, :target_time_label]) |>
        combine(findcoefs, __) |>
        insertcols!(__, :condition => "global-v-object") ),
    @_( classdf |> filter(_.condition in ["global","spatial"],__) |>
        groupby(__, [:hit, :salience_label, :target_time_label]) |>
        combine(findcoefs, __) |>
        insertcols!(__, :condition => "global-v-spatial") ),
    @_( classdf |> filter(_.condition in ["object","spatial"],__) |>
        groupby(__, [:hit, :salience_label, :target_time_label]) |>
        combine(findcoefs, __) |>
        insertcols!(__, :condition => "object-v-spatial") )
)

# do the classification accuracies look right

R"""
pl = ggplot(filter($coefs, hit == 'hit'),
        aes(x = target_time_label, y = correct, fill = salience_label)) +
    geom_bar(stat = 'identity', pos = position_dodge(width = 0.6), width = 0.6) +
    facet_wrap(~condition) +
    geom_hline(yintercept=0.5, linetype=2) +
    coord_cartesian(ylim=c(0.4,1))
"""

R"""
ggsave(file.path($dir,"logistic_correct.pdf"),pl,width=9,height=6)
"""

# sort of...??

# let's look a the coefficients

coefs_spread = @_ coefs |>
    stack(__, All(r"coef"), [:hit, :salience_label, :target_time_label,:condition],
        variable_name = :coef) |>
    transform!(__, :coef => ByRow(x -> parse(Int,match(r"coef([0-9]+)", x)[1])) => :coef) |>
    transform!(__, :coef => ByRow(x -> parsecoef(x)[1]) => :channel) |>
    transform!(__, :coef => ByRow(x -> parsecoef(x)[2]) => :freqbin)


coefs_spread_pl = @_ coefs_spread |> filter(_.hit == "hit",__)


R"""
df = $coefs_spread_pl
df$freqbin = factor(df$freqbin, levels = unique(df$freqbin), ordered = T)
df %>% arrange(freqbin)
pl = ggplot(df, aes(x = channel, y = value, fill = salience_label)) +
    geom_bar(stat='identity', pos = 'dodge') +
    xlab('MCCA Component') + ylab('Weight') +
    facet_grid(condition~target_time_label+freqbin, labeller = 'label_both')
"""

R"""
ggsave(file.path($dir,"logistic_coefs.pdf"),pl,width=11,height=8)
"""

# NOTE:
# - maybe z-score the features by frequency bin?
# - use a larger sparsity coefficient?
# - Both those things seemed to help; we need to check that this works#   from a cross-validation standpoint; re-run logistic with fixed C
#   using a cross-val approach
