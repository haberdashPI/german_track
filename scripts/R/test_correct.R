source("util/setup.R")

dir = file.path(plot_dir,paste("run",Sys.Date(),sep="_"))
dir.exists(dir) || dir.create(dir)

df = read.csv(file.path(cache_dir,"test_correct.csv"))
df = df %>% group_by(sid,condition,speaker) %>%
    mutate(trial = 1:length(corr)) %>%
    mutate(test_correct = test_correct == "true")

df %>% group_by(sid) %>% summarize(accuracy = mean(test_correct))

ggplot(df,aes(x=speaker,y=corr,color=test_correct)) +
    geom_point(position=position_jitterdodge(jitter.width=0.2,dodge.width=0.3),
        alpha=0.5,size=1,aes(shape=test_correct)) +
    stat_summary(geom="pointrange",fun.data="mean_cl_boot",size=0.3,
        position=position_nudge(x=0.4)) +
    geom_abline(slope=0,intercept=0,linetype=2) +
    scale_color_brewer(palette='Set1') +
    coord_cartesian(xlim=c(0.5,4.5)) +
    facet_grid(condition~sid)

ggsave(file.path(dir,"correct_corr_vs_control.pdf"),width=9,height=7)

dfcor = df %>%
    group_by(sid,condition,trial,test_correct) %>%
    spread(speaker,corr)

ggplot(dfcor,aes(x=pmax(fem1,fem2),y=male)) +
    geom_point(aes(color=test_correct)) +
    geom_abline(intercept=0,slope=1) + facet_grid(condition~sid) +
    scale_color_brewer(palette='Set1')
# ggsave(file.path(dir,"correct_corr_male_v_female_by_condition.pdf"),
#     width=9,height=5)

ggplot(dfcor,aes(x=pmax(fem1,fem2),y=male)) +
    geom_point(aes(color=test_correct)) +
    geom_abline(intercept=0,slope=1) + facet_grid(~sid) +
    scale_color_brewer(palette='Set1')
# ggsave(file.path(dir,"correct_corr_male_v_female.pdf"),width=9,height=2.5)
