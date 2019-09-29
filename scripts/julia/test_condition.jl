using DrWatson; quickactivate(@__DIR__,"german_track")
using GermanTrack

stim_info = JSON.parsefile(joinpath(stimulus_dir(),"config.json"))
eeg_files = filter(x -> occursin(r"_mcca34\.mcca_proj$",x),readdir(data_dir()))
eeg_files = eeg_files[1:1]

fbounds = trunc.(Int,round.(exp.(range(log(90),log(3700),length=5))[2:end-1],
    digits=-1))

encoding = JointEncoding(PitchSurpriseEncoding(),ASEnvelope())
# encoding = JointEncoding(PitchSurpriseEncoding(),ASBins(fbounds))

# eegencode = JointEncoding(
#     RawEncoding(),
#     FilteredPower("alpha",5,15),
#     FilteredPower("gamma",30,100)
# )

conditions = [
    string("all",cond) => @λ(_.condition == cond ? all_indices : no_indices)
    for cond in ["feature","object","test"]
]

df, = train_test(
    StaticMethod(NormL2(0.2),cor),
    SpeakerStimMethod(
        encoding=encoding,
        sources=["male","fem1","fem2","male_other"]),
    resample = 64,
    eeg_files,stim_info,
    train = conditions,
    # encode_eeg = eegencode
)
alert()

if :condition_str ∉ names(df)
    df[!,:condition_str] =  df.condition
end
df.condition = replace.(df.condition_str,Ref(r"train-all([[:alnum:]]+)_.*" => s"\1"))

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

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

ggsave(file.path($dir,"condition_test_pitch.pdf"),width=9,height=7)

"""

