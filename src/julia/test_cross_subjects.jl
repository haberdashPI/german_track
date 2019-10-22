using DrWatson; quickactivate(@__DIR__,"german_track"); using GermanTrack

stim_info = JSON.parsefile(joinpath(stimulus_dir(),"config.json"))
eeg_files = filter(x -> occursin(r"_mcca34\.mcca_proj$",x),readdir(data_dir()))

eeg_files = filter(@λ(!occursin("11",_)),eeg_files)

encoding = JointEncoding(PitchSurpriseEncoding(), ASEnvelope())

target_times =
    convert(Array{Float64},stim_info["test_block_cfg"]["target_times"])

before_target = map(target_times) do time
    iszero(time) ? no_indices : only_near(time,10,window=(-1.5,0))
end

const speakers = convert(Array{Int},
    stim_info["test_block_cfg"]["trial_target_speakers"])
const tindex = Dict("male" => 1, "fem" => 2)
const cond_label = Dict("object" => "object", "global" => "test", "spatial" => "feature")
const direction = convert(Array{String},
    stim_info["test_block_cfg"]["trial_target_dir"])

listen_conds = ["object","global", "spatial"]
targets = ["male","fem"]
labels = ["correct","all"]

conditions = Dict(
    (label=label,condition=condition,target=target) =>
        @λ(_row.condition == cond_label[condition] &&
           (label == "all" || _row.correct) &&
           speakers[_row.sound_index] == tindex[target] ?
                before_target[_row.sound_index] : no_indices)
    for condition in listen_conds
    for target in targets
    for label in labels
)

function measures(pred,stim)
    (joint_cor = cor(vec(pred),vec(stim)),
     male_cor = cor(vec(pred[:,1:2]),vec(stim[:,1:2])),
     fem1_cor = cor(vec(pred[:,3:4]),vec(stim[:,3:4])),
     fem2_cor = cor(vec(pred[:,5:6]),vec(stim[:,5:6])))
end

# the plan is to first look at the indices that are actually
# being trained and tested vs. the folds
df, models = train_test(
    K = 50,
    StaticMethod(NormL2(0.2),measures),
    SpeakerStimMethod(
        encoding=encoding,
        sources=[joint_source, other(joint_source)]),
    resample = 64,
    eeg_files, stim_info,
    maxlag=0.8,
    return_models = true,
    train = subdict(conditions,
        (label = "correct", condition = cond, target = target)
        for cond in listen_conds, target in targets
    ),
    test = subdict(conditions,
        (label = "all", condition = cond, target = target)
        for cond in listen_conds, target in targets
    )
);
alert()

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
    facet_grid(train_condition+test_condition~sid+source,labeller=label_context) +
    geom_abline(slope=0,intercept=0,linetype=2)

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

ggsave(file.path($dir,"cross_subjects.pdf"),width=11,height=8)

"""
