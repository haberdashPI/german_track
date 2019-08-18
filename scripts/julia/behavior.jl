
using DrWatson; quickactivate(@__DIR__,"german_track")
include(joinpath(srcdir(),"julia","setup.jl"))

stim_info = JSON.parsefile(joinpath(stimulus_dir,"config.json"))
eeg_files = filter(x -> occursin(r"_mcca65\.bson$",x),readdir(data_dir))

df = DataFrame()
for file in eeg_files
    global df
    df_, sid = events_for_eeg(file,stim_info)
    df_[!,:sid] .= sid
    df = vcat(df,df_)
end

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

R"""

library(ggplot2)
library(dplyr)

dprime = function(hits,falarm){
    qnorm((sum(hits)+1)/(length(hits)+1)) -
        qnorm((sum(falarm)+1)/(length(falarm)+1))
}

summary = $df %>% group_by(sid,condition) %>%
    summarize(dp = dprime(target_present & correct,!target_present & !correct),
              mean = mean(correct))

ggplot(summary,aes(x=condition,y=mean)) +
  geom_point(aes(group=sid),position=position_dodge(width=0.2)) +
  stat_summary()

ggplot(summary,aes(x=condition,y=dp)) +
  geom_point(aes(group=sid),position=position_dodge(width=0.2)) +
  stat_summary()

ggsave(file.path($dir,"behavior.pdf"),width=6,height=4)

# show buildup? (dprime as a function of time)
summary = $df %>% group_by(sid,condition) %>%
    do()
"""
