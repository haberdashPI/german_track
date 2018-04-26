library(dplyr)
library(ggplot2)
library(tidyr)

source('util/setup.R')

dir = file.path(plot_dir,paste('run',Sys.Date(),sep='_'))
dir.exists(dir) || dir.create(dir)

df = NULL
## cor_files = list.files(cache_dir,pattern='target_2.0seconds.*_cor.csv')
cor_files = list.files(cache_dir,pattern='^target_model_.*cor.csv')
for(file in cor_files){
  dff = read.csv(file.path(cache_dir,file))

  dff = dff %>%
    mutate(target_heard = response == 2,
           target_present = !is.na(target_time) & target_time > 0,
           hit = target_heard & target_present,
           miss = !target_heard & target_present) %>%
    arrange(time)

  df = rbind(df,dff)
}

dfview = df %>% filter(!is.na(hit))
## ggplot(dfview,aes(x=hit,y=cor)) +
##   facet_grid(condition~model) +
##   geom_point(alpha=0.5,position=position_jitter(width=0.05)) +
##   stat_summary(aes(x=as.numeric(hit)+1.2)) +
##     geom_hline(yintercept=0,linetype=2)

dfmodels = dfview %>%
  select(-test) %>%
  spread(model,cor)

ggplot(dfmodels,aes(x=hit,y=hit_target-miss_nontargets)) +
  facet_grid(condition~sid) +
  geom_point(alpha=0.5,position=position_jitter(width=0.05)) +
  stat_summary(aes(x=as.numeric(hit)+1.2)) +
  geom_hline(yintercept=0,linetype=2)

ggplot(dfmodels,aes(x=hit,y=hit_target-miss_nontargets)) +
  geom_point(alpha=0.5,position=position_jitter(width=0.05)) +
  stat_summary(aes(x=as.numeric(hit)+1.2)) +
  geom_hline(yintercept=0,linetype=2)

t.test(with(subset(dfmodels,hit),hit_target-miss_nontargets),
       with(subset(dfmodels,!hit),hit_target-miss_nontargets))

t.test(with(subset(dfmodels,hit & condition == 'test'),hit_target-miss_nontargets),
       with(subset(dfmodels,!hit),hit_target-miss_nontargets))

t.test(with(subset(dfmodels,hit & condition == 'feature'),hit_target-miss_nontargets),
       with(subset(dfmodels,!hit),hit_target-miss_nontargets))

t.test(with(subset(dfmodels,hit & condition == 'object'),hit_target-miss_nontargets),
       with(subset(dfmodels,!hit),hit_target-miss_nontargets))
