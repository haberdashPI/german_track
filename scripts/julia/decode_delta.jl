using DrWatson; @quickactivate("german_track")
include(joinpath(srcdir(), "julia", "setup.jl"))

# eeg_files = filter(x->occursin(r"_mcca03\.mcca_proj$", x), readdir(data_dir()))
eeg_files = filter(x->occursin(r"_mcca34\.mcca_proj$", x), readdir(data_dir()))
# eeg_files = filter(x->occursin(r"_cleaned\.eeg$", x), readdir(data_dir()))

eeg_encoding = FFTFiltered("delta" => (1.0,3.0),seconds=10,fs=10,nchannels=34)
encoding = JointEncoding(PitchSurpriseEncoding(), ASEnvelope())

import GermanTrack: stim_info, speakers, directions, target_times, switch_times

cachefile = joinpath(cache_dir(),"..","subject_cache","delta_subjects.bson")
if isfile(cachefile)
    @load cachefile subjecst
else
    subjects = Dict(file =>
        load_subject(joinpath(data_dir(), file),
            stim_info,
            encoding = eeg_encoding,
            framerate=10)
        for file in eeg_files)
    @save cachefile subjects
end

const tindex = Dict("male" => 1, "fem" => 2)

before_target = map(target_times) do time
    iszero(time) ? no_indices : only_near(time,10,window=(-1.5,0))
end

listen_conds = ["object","global"]
sids = getproperty.(values(subjects),:sid)
condition_targets = Dict("object" => [1], "global" => [1,2])

conditions = Dict(
    (sid=sid,label=label,condition=condition) =>
        @λ(_row.condition == condition &&
           (_row.sid == sid) &&
           (label == "all" || _row.correct) &&
           (speakers[_row.sound_index] ∈ condition_targets[condition]) ?
                before_target[_row.sound_index] : no_indices)
    for sid in sids
    for condition in listen_conds
    for label in ["correct", "all"]
)

# the plan is to first look at the indices that are actually
# being trained and tested vs. the folds
df = train_test(
    StaticMethod(NormL2(0.2),cor),
    SpeakerStimMethod(
        encoding=encoding,
        sources=[male_source,fem1_source,fem2_source,mixed_sources,
                 fem_mix_sources,joint_source,other(male_source)]),
    subjects = subjects,
    encode_eeg = eeg_encoding,
    resample = 10,
    eeg_files, stim_info,
    maxlag=0.8,
    train = subdict(conditions,
        (sid = sid, label = "correct", condition = cond)
        for cond in listen_conds, sid in sids
    ),
    test = subdict(conditions,
        (sid = sid, label = "all", condition = cond)
        for cond in listen_conds, sid in sids
    )
);
alert()

# df[!,:location] = directions[df.stim_id]

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

R"""

library(tidyr)
library(dplyr)
library(ggplot2)
library(stringr)

ggplot($df,aes(x=source,y=value,color=test_correct)) +
    stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
        position=position_nudge(x=0.3)) +
    geom_point(alpha=0.5,position=position_jitter(width=0.1)) +
    scale_color_brewer(palette='Set1') +
    facet_grid(train_condition~sid,labeller=label_context)

group_mean = $(df) %>%
    group_by(sid,source,test_correct,train_condition) %>%
    summarize(value=mean(value))

ggplot(group_mean,aes(x=source,y=value,color=test_correct)) +
    stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
        position=position_nudge(x=0.3)) +
    geom_point(alpha=0.5,position=position_jitter(width=0.1)) +
    scale_color_brewer(palette='Set1') +
    facet_grid(train_condition~.,labeller=label_context)

"""
