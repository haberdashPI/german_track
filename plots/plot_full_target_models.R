library(dplyr)
library(ggplot2)
library(tidyr)

source('util/setup.R')

dir = file.path(plot_dir,paste('run',Sys.Date(),sep='_'))
dir.exists(dir) || dir.create(dir)

df = NULL
cor_files = list.files(cache_dir,pattern='full_target.*_cor.csv')
for(file in cor_files){
  dff = read.csv(file.path(cache_dir,file)) %>%
    mutate(condition = recode(condition,test = 'globalc'),
           target_heard = response == 2,
           target_present = !is.na(target_time) & target_time > 0,
           hit = target_heard & target_present,
           miss = !target_heard & target_present) %>%
    arrange(time)

  df = rbind(df,dff)
}

dfview = df %>% filter(sid == 2,!is.na(hit))

dftests = dfview %>%
    spread(test,cor)

ggplot(dftests,aes(x=hit,y=target-mixture)) +
  facet_grid(condition~.) +
  geom_point(alpha=0.5,position=position_jitter(width=0.05)) +
  stat_summary(aes(x=as.numeric(hit)+1.2)) +
  geom_hline(yintercept=0,linetype=2)
