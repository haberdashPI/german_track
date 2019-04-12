source("util/setup.R")

dir = file.path(plot_dir,paste("run",Sys.Date(),sep="_"))
dir.exists(dir) || dir.create(dir)

df = read.csv(file.path(cache_dir,"testcond.csv"))
df = df %>% group_by(sid,condition,speaker) %>%
    mutate(trial = 1:length(corr))

ggplot(df,aes(x=speaker,y=corr,color=speaker)) +
    geom_point(position=position_jitter(width=0.2),color="black",alpha=0.2) +
    stat_summary(geom="pointrange",fun.data="mean_cl_boot") +
    geom_abline(slope=0,intercept=0,linetype=2) +
    facet_wrap(condition~sid)
ggsave(file.path(dir,"corr_vs_control.pdf"))

# TODO: reshape columns
# plot vs. correlations

dfcor = df %>% group_by(sid,condition,trial) %>% spread(speaker,corr)

ggplot(dfcor,aes(x=pmax(fem1,fem2),y=male)) + geom_point() +
    geom_abline(intercept=0,slope=1) + facet_wrap(condition~sid)

ggplot(dfcor,aes(x=other_male,y=male)) + geom_point() +
    geom_abline(intercept=0,slope=1) + facet_wrap(condition~sid) +
    theme_classic()
