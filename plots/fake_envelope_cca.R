library(dplyr)
library(stringr)
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

dfview = df %>%
  filter(sid == 2) %>%
  gather(target,cor,cor_object:cor_test) %>%
  mutate(target = str_sub(target,5,-1))
ggplot(dfview,aes(x=target,y=cor)) + geom_point() + facet_wrap(~condition)
