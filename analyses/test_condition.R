source("util/setup.R")

dir = file.path(plot_dir,paste("run",Sys.Date(),sep="_"))
dir.exists(dir) || dir.create(dir)

df = read.csv(file.path(cache_dir,"testcond.csv"))

ggplot(df,aes(x=speaker,y=corr,color=speaker)) +
    geom_point(position=position_jitter(width=0.2),color="black",alpha=0.2) +
    stat_summary(geom="pointrange",fun.data="mean_cl_boot") +
    geom_abline(slope=0,intercept=0,linetype=2) +
    facet_wrap(condition~sid)

ggsave(file.path(dir,"conditions.pdf"))
