using DrWatson; quickactivate(@__DIR__,"german_track")
include(joinpath(srcdir(),"julia","setup.jl"))

# - train at correct targets

stim_info = JSON.parsefile(joinpath(stimulus_dir(),"config.json"))
eeg_files = filter(x -> occursin(r"_mcca34\.mcca_proj$",x),readdir(data_dir()))
# eeg_files = filter(x -> occursin(r"_mcca65\.bson$",x),readdir(data_dir()))

# TODO: we don't need this file format, we can use the 65 components directly,
# to reduce memory load.

df = train_stimuli(
    StaticMethod(),
    SpeakerStimMethod(encoding=ASEnvelope()),
    resample = 64,
    eeg_files,stim_info,
    train = "correct" =>
        row -> row.correct ? all_indices : no_indices,
    test = "testcorrect" => all_indices,
    skip_bad_trials = true,
)
alert()


dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

R"""

library(dplyr)
library(ggplot2)

df = $df %>% group_by(sid,condition,source) %>%
    mutate(trial = 1:length(corr))

df %>% group_by(sid) %>% summarize(accuracy = mean(test_correct))

ggplot(df,aes(x=source,y=corr,color=test_correct)) +
    geom_point(position=position_jitter(width=0.1),
        alpha=0.5,size=0.8) +
    stat_summary(geom="pointrange",fun.data="mean_cl_boot",size=0.2,
        position=position_nudge(x=0.3)) +
    geom_abline(slope=0,intercept=0,linetype=2) +
    scale_color_brewer(palette='Set1') +
    coord_cartesian(xlim=c(0.5,5.5)) +
    facet_grid(condition~sid)

ggsave(file.path($dir,"test_correct_new.pdf"),width=9,height=7)

"""

