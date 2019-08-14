include(joinpath(@__DIR__,"..","util","setup.jl"))

# - train on correct trials only
# - train at target switches

stim_info = JSON.parsefile(joinpath(stimulus_dir,"config.json"))
eeg_files = filter(x -> occursin(r"_mcca65\.bson$",x),readdir(data_dir))

# TODO: we don't need this file format, we can use the 65 components directly,
# to reduce memory load.

df = train_stimuli(
    StaticMethod(),
    SpeakerStimMethod(envelope_method=:audiospect),
    eeg_files,stim_info,
    train = "all" => all_indices,
    skip_bad_trials = true
)

dir = joinpath(plot_dir,string("results_",Date(now())))
isdir(dir) || mkdir(dir)
R"""

library(dplyr)
library(ggplot2)

df = $df %>% group_by(sid,condition,source) %>%
    mutate(trial = 1:length(corr))

ggplot(df,aes(x=source,y=corr,color=source)) +
    geom_point(position=position_jitter(width=0.1),
        alpha=0.5,size=0.8) +
    stat_summary(geom="pointrange",fun.data="mean_cl_boot",size=0.2,
        position=position_nudge(x=0.3)) +
    geom_abline(slope=0,intercept=0,linetype=2) +
    scale_color_brewer(palette='Set1') +
    coord_cartesian(xlim=c(0.5,5.5)) +
    facet_grid(condition~sid)

ggsave(file.path($dir,"by_condition.pdf"),width=9,height=7)

"""

save(joinpath(cache_dir(),"test_condition_rms.csv"),df)

alert()
