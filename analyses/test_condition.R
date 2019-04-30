source("util/setup.R")

dir = file.path(plot_dir,paste("run",Sys.Date(),sep="_"))
dir.exists(dir) || dir.create(dir)

df = read.csv(file.path(cache_dir,"test_condition.csv"))
df = df %>% group_by(sid,condition,speaker) %>%
    mutate(trial = 1:length(corr))

ggplot(df,aes(x=speaker,y=corr,color=speaker)) +
    geom_point(position=position_jitter(width=0.1),
        alpha=0.5,size=0.8) +
    stat_summary(geom="pointrange",fun.data="mean_cl_boot",size=0.2,
        position=position_nudge(x=0.3)) +
    geom_abline(slope=0,intercept=0,linetype=2) +
    scale_color_brewer(palette='Set1') +
    coord_cartesian(xlim=c(0.5,4.5)) +
    facet_grid(condition~sid)

ggsave(file.path(dir,"all_corr_vs_control.pdf"),width=9,height=7)

# TODO: reshape columns
# plot vs. correlations

dfcor = df %>% group_by(sid,condition,trial) %>% spread(speaker,corr)

ggplot(dfcor,aes(x=pmax(fem1,fem2),y=male)) + geom_point() +
    geom_abline(intercept=0,slope=1) + facet_grid(condition~sid)
# ggsave(file.path(dir,"all_corr_male_v_female_by_condition.pdf"),
#     width=9,height=5)

ggplot(dfcor,aes(x=pmax(fem1,fem2),y=male)) + geom_point() +
    geom_abline(intercept=0,slope=1) + facet_grid(~sid)
# ggsave(file.path(dir,"all_corr_male_v_female.pdf"),width=9,height=2.5)
