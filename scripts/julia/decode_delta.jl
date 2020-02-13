using DrWatson; @quickactivate("german_track")
include(joinpath(srcdir(), "julia", "setup.jl"))

# eeg_files = filter(x->occursin(r"_mcca03\.mcca_proj$", x), readdir(data_dir()))
eeg_files = filter(x->occursin(r"_mcca34\.mcca_proj$", x), readdir(data_dir()))
# eeg_files = filter(x->occursin(r"_cleaned\.eeg$", x), readdir(data_dir()))

eeg_encoding = JointEncoding(FFTFiltered())
encoding = JointEncoding(PitchSurpriseEncoding(), ASEnvelope())

import GermanTrack: stim_info, speakers, directions, target_times, switch_times
subjects = Dict(file => load_subject(joinpath(data_dir(), file), stim_info,
                                     encoding = eeg_encoding)
    for file in eeg_files)
const tindex = Dict("male" => 1, "fem" => 2)

before_target = map(target_times) do time
    iszero(time) ? no_indices : only_near(time,10,window=(-1.5,0))
end

conditions = Dict(
    (sid=sid,label=label,condition=condition,target=target) =>
        @λ(_row.condition == cond_label[condition] &&
           (_row.sid == sid) &&
           (label == "all" || _row.correct) &&
           speakers[_row.sound_index] == tindex[target] ?
                before_target[_row.sound_index] : no_indices)
    for sid in sidfor.(eeg_files)
    for condition in listen_conds
    for target in targets
    for label in labels
)

# the plan is to first look at the indices that are actually
# being trained and tested vs. the folds
listen_conds = first(subjects)[2].events.condition |> unique
cond_pairs = Iterators.product(listen_conds,listen_conds)
df = train_test(
    StaticMethod(NormL2(0.2),cor),
    SpeakerStimMethod(
        encoding=encoding,
        sources=[male_source,fem1_source,fem2_source,mixed_sources,
                 fem_mix_sources,joint_sources,other(male)]),
    encode_eeg = eeg_encoding,
    resample = 64,
    eeg_files, stim_info,
    maxlag=0.8,
    train = subdict(conditions,
        (sid = sid, label = "correct", condition = cond, target = target)
        for (cond,_) in cond_pairs, target in targets, sid in sidfor.(eeg_files)
    ),
    test = subdict(conditions,
        (sid = sid, label = "all", condition = cond, target = target)
        for (_,cond) in cond_pairs, target in targets, sid in sidfor.(eeg_files)
    )
);
alert()

df[!,:location] = direction[df.stim_id]

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

R"""

library(tidyr)
library(dplyr)
library(ggplot2)
library(stringr)

ggplot($df,aes(x=test_target,y=joint_cor,color=test_correct)) +
    stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
        position=position_nudge(x=0.3)) +
    geom_point(alpha=0.5,position=position_jitter(width=0.1)) +
    scale_color_brewer(palette='Set1') +
    facet_grid(train_condition+test_condition~sid+source,labeller=label_context)

dfmatch = $df %>% filter(source == 'joint', train_condition == test_condition) %>%
    rename(condition = test_condition, target = test_target,
        target_detected = test_correct) %>%
    group_by(sid,target_detected,target,condition,stim_id) %>%
    gather(male_cor,fem1_cor,fem2_cor,key='featuresof',value='cor') %>%
    mutate(featuresof = str_replace(featuresof,"(.*)_cor","\\1"))

pos = position_jitterdodge(jitter.width=0.1,dodge.width=0.3)
ggplot(dfmatch,aes(x=featuresof,y=cor,color=target_detected)) +
    stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
        aes(fill=target_detected),pch=21,size=0.5,
        color='black',
        position=position_dodge(width=0.75)) +
    geom_point(alpha=0.5,position= pos) +
    scale_color_brewer(palette='Set1') +
    scale_fill_brewer(palette='Set1') +
    theme_classic() +
    facet_grid(condition~sid+target,labeller=label_context) +
    geom_abline(intercept=0,slope=0,linetype=2)

ggsave(file.path($dir,"within_condition.pdf"),width=11,height=8)

ggplot(dfmatch,aes(x=target_detected,y=cor,color=target_detected)) +
    stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
        position=position_dodge(width=0.85)) +
    scale_color_brewer(palette='Set1')

ggplot(dfmatch,aes(x=featuresof,y=cor,color=interaction(location,target_detected))) +
    stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
        aes(fill=interaction(location,target_detected)),pch=21,size=0.5,
        color='black',
        position=position_nudge(x=0.3)) +
    geom_point(alpha=0.5,position=position_jitter(width=0.1)) +
    scale_color_brewer(palette='Paired') +
    scale_fill_brewer(palette='Paired') +
    theme_classic() +
    facet_grid(condition~sid+target,labeller=label_context)

ggsave(file.path($dir,"test_across_conditions_spatial.pdf"))
"""

matched = @where(models,
    (:train_condition .== :test_condition) .&
    (:source .== "joint"))

function bootstrap(df,fn,col;conf=0.95,N=10_000)
    # bootstrap across the trials
    med = similar(fn(df[1,col]))
    lower = similar(med)
    upper = similar(med)

    M = size(df,1)
    baserng = MersenneTwister()
    alldata = fn.(df[:,col])
    for i in eachindex(med)
        rng = copy(baserng)
        samples = map(1:N) do _
            mean(alldata[j][i] for j in sample(1:M,M))
        end
        med[i], lower[i], upper[i] =
            quantile(samples,[0.5,conf/2,1-(conf/2)])
    end

    med, lower, upper
end

feature_names = [
    string(feature,"_",source)
    for feature = [:envelop,:pitch],
        source = [:male,:fem1,:fem2]
]

bootlag = by(matched,[:source,:sid,:test_target,:train_condition]) do trials
    result = bootstrap(trials,@λ(mean(_,dims=1)),:coefs)
    function todf(coefs)
        coefs, = PlotAxes.asplotable(coefs,:col,:page)
        rename!(coefs,:col => :lag, :page => :feature)
        coefs[!,:feature] .= feature_names[convert.(Int,coefs.feature)]
        coefs
    end

    med, lower, upper = todf.(result)
    med[!,:lower] = lower.value
    med[!,:upper] = upper.value

    med
end

R"""

ggplot($bootlag,aes(x=lag,y=value,group=train_condition)) +
    facet_grid(feature~sid+test_target) +
    geom_errorbar(aes(ymin=lower,ymax=upper,color=train_condition),size=0.5,width=0.25) +
    geom_point(aes(color=train_condition),size=0.05) +
    geom_line(aes(color=train_condition))

ggsave(file.path($dir,"lags.pdf"),width=11,height=8)
"""

bootcomp = by(matched,[:source,:sid,:test_target,:train_condition]) do trials
    result = bootstrap(trials,@λ(mean(_,dims=2)),:coefs)
    function todf(coefs)
        coefs, = PlotAxes.asplotable(coefs,:row,:page)
        rename!(coefs,:row => :component, :page => :feature)
        coefs[!,:feature] .= feature_names[convert.(Int,coefs.feature)]
        coefs
    end

    med, lower, upper = todf.(result)
    med[!,:lower] = lower.value
    med[!,:upper] = upper.value

    med
end

R"""

ggplot($bootcomp,aes(x=component,y=value,group=train_condition)) +
    facet_grid(feature~sid+test_target) +
    geom_errorbar(aes(ymin=lower,ymax=upper,color=train_condition)) +
    geom_point(aes(color=train_condition),size=0.25) + coord_flip() +
    geom_line(aes(color=train_condition))

ggsave(file.path($dir,"components.pdf"),width=8,height=11)

"""


# TODO: handle bootstrapping
# # TODO: generalize this code and make it part of `train_test`
# # quant = (10,10,10)
# all_coefs = mapreduce(vcat,eachrow(decoders)) do row
#     coefs, = PlotAxes.asplotable(row.coefs, quantize = (100,100,10))
#     rename!(coefs,:row => :component, :col => :lag, :page => :feature)
#     coefs[!,:feature] .= feature_names[coefs.feature]

#     for col in setdiff(names(row),[:coefs])
#         coefs[!,col] .= row[col]
#     end
#     coefs
# end
# all_coefs = addconds!(all_coefs)


# R"""

# df = filter($all_coefs,
#     test_condition == train_condition,
#     (train_target == 'before_correct_male') == (test_target == 'male'))

# for(sid_ in 8:14){

#     ggplot(filter(df,sid == sid_),aes(y=component,x=lag,fill=value)) +
#         geom_raster() +
#         facet_grid(feature~train_condition+test_target) +
#         scale_fill_distiller(palette='RdBu')
#     ggsave(file.path($dir,sprintf("global_v_object_coefs_sid_%02d.pdf",sid_)))

#     dflags = df %>%
#         group_by(feature,train_condition,test_target,sid,lag) %>%
#         summarize(value = mean(value))

#     ggplot(file.path($dir,filter(dflags,sid == sid_),aes(y=value,x=lag,color=train_condition))) +
#         geom_line() +
#         facet_grid(feature~test_target) +
#         scale_fill_distiller(palette='RdBu')
#         ggsave(sprintf("global_v_object_lags_sid_%02d.pdf",sid_))

#     dfcomps = df %>%
#         group_by(feature,train_condition,test_target,sid,component) %>%
#         summarize(value = mean(value))

#     ggplot(filter(dfcomps,sid == sid_),aes(y=value,x=component,color=train_condition)) +
#         geom_line() + coord_flip() +
#         facet_grid(feature~test_target) +
#         scale_fill_distiller(palette='RdBu')
#         ggsave(file.path($dir,sprintf("global_v_object_components_sid_%02d.pdf",sid_)))
# }

# """
