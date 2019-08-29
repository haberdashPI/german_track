
using DrWatson; quickactivate(@__DIR__,"german_track")
using GermanTrack

stim_info = JSON.parsefile(joinpath(stimulus_dir(),"config.json"))
eeg_files = filter(x -> occursin(r"_mcca34\.mcca_proj$",x),readdir(data_dir()))
eeg_files = eeg_files[1:1]

encoding = JointEncoding(PitchSurpriseEncoding(),ASEnvelope())

fs = convert(Float64,stim_info["fs"])
switch_times =
    convert(Array{Array{Float64}},stim_info["test_block_cfg"]["switch_times"])
switch_bounds = only_near.(map(x -> x./fs,switch_times),10)

# TODO:
# 2 problmes
# wrong sample rate for input (??)
# wrong time slices

df = train_stimuli(
    StaticMethod(NormL2(0.2),cor),
    SpeakerStimMethod(encoding=encoding),
    eeg_files, stim_info,
    resample = 64,
    train = "switch_only" => row -> switch_bounds[row.sound_index],
    test = "clean_switch_only" => row -> switch_bounds[row.sound_index],
    skip_bad_trials=true)
alert()

R"""

library(dplyr)
library(ggplot2)

df = $df %>% group_by(sid,condition,source) %>%
    mutate(trial = 1:length(value))

ggplot(df,aes(x=source,y=value,color=source)) +
    geom_point(position=position_jitter(width=0.1),
        alpha=0.5,size=0.8) +
    stat_summary(geom="pointrange",fun.data="mean_cl_boot",size=0.2,
        position=position_nudge(x=0.3)) +
    geom_abline(slope=0,intercept=0,linetype=2) +
    scale_color_brewer(palette='Set1') +
    coord_cartesian(xlim=c(0.5,5.5)) +
    facet_grid(condition~sid)

ggsave(file.path($dir,"switches.pdf"),width=9,height=7)

"""


