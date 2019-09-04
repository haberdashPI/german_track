
# can we tell a difference between decoding far from target and decoding close
# to targets does this depend on whether the participant responded correctly

using DrWatson; quickactivate(@__DIR__,"german_track")
using GermanTrack

stim_info = JSON.parsefile(joinpath(stimulus_dir(),"config.json"))
eeg_files = filter(x -> occursin(r"_mcca34\.mcca_proj$",x),readdir(data_dir()))
eeg_files = eeg_files[1:1]

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

conds_during_target = [
    string("during_",cond,"_","target") =>
        @λ(_row.condition == cond ? during_target[_row.sound_index] : no_indices)
    for cond in ["feature","object","test"]
]
conds_before_target = [
    string("before_",cond,"_","target") =>
        @λ(_row.condition == cond ? before_target[_row.sound_index] : no_indices)
    for cond in ["feature","object","test"]
]

df = train_stimuli(
    StaticMethod(NormL2(0.2),cor),
    SpeakerStimMethod(encoding=encoding),
    resample = 64,
    eeg_files, stim_info,
    train = [conds_during_target;conds_during_target],
    test = [conds_during_target;conds_before_target],
    skip_bad_trials = true,
)
alert()

df[!,:test] = replace.(df.condition,Ref(r"train.*_target_test([a-z]+)_.*_target" => s"\1"))
df.condition = replace.(df.condition,Ref(r"trainduring_([a-z]+)_target_test.*" => s"\1"))

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

R"""

library(tidyr)
library(dplyr)
library(ggplot2)

df = $df

ggplot(df,aes(x=test,y=value,color=source)) +
    geom_point(position=position_jitter(width=0.1),
        alpha=0.5,size=0.8) +
    stat_summary(geom="pointrange",fun.data="mean_cl_boot",size=0.2,
        position=position_nudge(x=0.3)) +
    geom_abline(slope=0,intercept=0,linetype=2) +
    scale_color_brewer(palette='Set1') +
    coord_cartesian(xlim=c(0.5,5.5)) +
    facet_grid(condition~sid+source)

dfcor = df %>%
    group_by(sid,condition,trial,test_correct) %>%
    spread(test,value)

ggplot(dfcor,aes(x=before_target,y=during_target,color=source)) +
    geom_point(alpha=0.5) +
    geom_abline(slope=1,intercept=0,linetype=2) +
    scale_color_brewer(palette='Set1') +
    facet_grid(condition~sid+source)

ggsave(file.path($dir,"before_after_target.pdf"),width=9,height=7)

ggplot(dfcor,aes(x=before_target,y=during_target,color=test_correct)) +
    geom_point(alpha=0.5) +
    geom_abline(slope=1,intercept=0,linetype=2) +
    scale_color_brewer(palette='Set1') +
    facet_grid(condition~sid+source)

ggsave(file.path($dir,"before_after_target_correct_untrained.pdf"),width=9,height=7)

"""

# df1 = train_stimuli(
#     StaticMethod(NormL2(0.2),cor),
#     SpeakerStimMethod(encoding=encoding),
#     resample = 64,
#     eeg_files, stim_info,
#     # train = "all" => all_indices,
#     train = "during_correct_target" =>
#         row -> row.correct ? during_target[row.sound_index] :
#             no_indices,
#     test = "during_target" =>
#         row -> during_target[row.sound_index],
#     skip_bad_trials = true,
# )
# alert()

# df1[!,:test] .= "during_target"

# df2 = train_stimuli(
#     StaticMethod(NormL2(0.2),cor),
#     SpeakerStimMethod(encoding=encoding),
#     resample = 64,
#     eeg_files, stim_info,
#     train = "during_correct_target" => row -> during_target[row.sound_index],
#     test = "before_target" => row -> before_target[row.sound_index],
#     skip_bad_trials = true,
# )
# df2[!,:test] .= "before_target"

# dfcorrect = vcat(df1,df2)
# categorical!(df,:test)

# R"""

# df = $dfcorrect

# ggplot(df,aes(x=test,y=value,color=source)) +
#     geom_point(position=position_jitter(width=0.1),
#         alpha=0.5,size=0.8) +
#     stat_summary(geom="pointrange",fun.data="mean_cl_boot",size=0.2,
#         position=position_nudge(x=0.3)) +
#     geom_abline(slope=0,intercept=0,linetype=2) +
#     scale_color_brewer(palette='Set1') +
#     coord_cartesian(xlim=c(0.5,5.5)) +
#     facet_grid(condition~sid+source)

# dfcor = df %>%
#     group_by(sid,condition,trial,test_correct) %>%
#     spread(test,value)

# ggplot(dfcor,aes(x=before_target,y=during_target,color=source)) +
#     geom_point(alpha=0.5) +
#     geom_abline(slope=1,intercept=0,linetype=2) +
#     scale_color_brewer(palette='Set1') +
#     facet_grid(condition~sid+source)

# ggsave(file.path($dir,"before_after_correct_target.pdf"),width=9,height=7)

# ggplot(dfcor,aes(x=before_target,y=during_target,color=test_correct)) +
#     geom_point(alpha=0.5) +
#     geom_abline(slope=1,intercept=0,linetype=2) +
#     scale_color_brewer(palette='Set1') +
#     facet_grid(condition~sid+source)

# ggsave(file.path($dir,"before_after_correct_target_by_correct.pdf"),width=9,height=7)

# """

