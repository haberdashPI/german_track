
# can we tell a difference between decoding far from target and decoding close
# to targets does this depend on whether the participant responded correctly

using DrWatson; quickactivate(@__DIR__,"german_track")
using GermanTrack

stim_info = JSON.parsefile(joinpath(stimulus_dir(),"config.json"))
eeg_files = filter(x -> occursin(r"_mcca34\.mcca_proj$",x),readdir(data_dir()))
# eeg_files = eeg_files[1:1]

encoding = JointEncoding(PitchSurpriseEncoding(),ASEnvelope())

target_times =
    convert(Array{Float64},stim_info["test_block_cfg"]["target_times"])

during_target = map(target_times) do time
    iszero(time) ? no_indices : only_near(time,10,window=(0,1.5))
end
before_target = map(target_times) do time
    iszero(time) ? no_indices : only_near(time,10,window=(-1.5,0))
end

# TODO: get the below to work

conds_during_correct_target = [
    string("during_correct_",cond,"_","target") =>
        @λ(_row.condition == cond && _row.correct ?
            during_target[_row.sound_index] : no_indices)
    for cond in ["feature","object","test"]
]
conds_during_target = [
    string("during_",cond,"_","target") =>
        @λ(_row.condition == cond ?
            during_target[_row.sound_index] : no_indices)
    for cond in ["feature","object","test"]
]
conds_before_target = [
    string("before_",cond,"_","target") =>
        @λ(_row.condition == cond ?
            before_target[_row.sound_index] : no_indices)
    for cond in ["feature","object","test"]
]

df = train_test(
    StaticMethod(NormL2(0.2),cor),
    SpeakerStimMethod(
        encoding=encoding,
        sources=["all-male","male","fem1","fem2","male_other"]),
    resample = 64,
    eeg_files, stim_info,
    train = [conds_during_target;conds_during_target;
             conds_during_correct_target;conds_during_correct_target],
    test = [conds_during_target;conds_before_target;
            conds_during_target;conds_before_target],
    skip_bad_trials = true,
)
alert()

df[!,:condition_str] = df.condition
df[!,:test] = replace.(df.condition_str,
    Ref(r"^.*test-([a-z]+)_.*$" => s"\1"))
df[!,:train_correct] = occursin.("correct",df.condition_str)
df.condition = replace.(df.condition_str,
    Ref(r"^.*train-during(_correct)?_([a-z]+)_.*$" => s"\2"))

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

R"""

library(tidyr)
library(dplyr)
library(ggplot2)

dfcor = $df %>%
    select(-condition_str) %>%
    group_by(trial,test_correct,train_correct,source,sid,condition) %>%
    spread(test,value) %>%
    ungroup()

ggplot(filter(dfcor,!test_correct & source != "all-male"),
    aes(x=before,y=during,color=source)) +
    geom_point(alpha=0.5) +
    geom_abline(slope=1,intercept=0,linetype=2) +
    scale_color_brewer(palette='Set1') +
    facet_grid(sid~condition+source)

ggsave(file.path($dir,"before_after_target.pdf"),width=9,height=7)

ggplot(filter(dfcor,!train_correct & source != "all-male"),
    aes(x=before,y=during,color=test_correct)) +
    geom_point(alpha=0.5) +
    geom_abline(slope=1,intercept=0,linetype=2) +
    scale_color_brewer(palette='Set1') +
    facet_grid(sid~condition+source)

ggplot(filter(dfcor,train_correct & source != "all-male"),
    aes(x=before,y=during,color=test_correct)) +
    geom_point(alpha=0.5) +
    geom_abline(slope=1,intercept=0,linetype=2) +
    scale_color_brewer(palette='Set1') +
    facet_grid(sid~condition+source)

ggsave(file.path($dir,"before_after_target_correct.pdf"),width=9,height=7)

"""
