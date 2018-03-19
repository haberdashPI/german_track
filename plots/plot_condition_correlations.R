library(dplyr)
library(ggplot2)
library(tidyr)

source('util/setup.R')

dir = file.path(plot_dir,paste('run',Sys.Date(),sep='_'))
dir.exists(dir) || dir.create(dir)

df = NULL
cor_files = list.files(cache_dir,pattern='condition.*_cor.csv')
for(file in cor_files){
  dff = read.csv(file.path(cache_dir,file))

  dff = dff %>%
    rename(globalcor = test_condition_cor,
           objectcor = object_condition_cor,
           featurecor = feature_condition_cor) %>%
    mutate(condition = recode(condition,test = 'globalc'),
           target_heard = response == 2,
           target_present = !is.na(target_time) & target_time > 0,
           hit = target_heard & target_present,
           miss = !target_heard & target_present) %>%
    arrange(time)

  df = rbind(df,dff)
}

dflong = df %>%
  gather(condition_model,cor,globalcor,objectcor,featurecor) %>%
  mutate(condition_model = factor(condition_model),
         condition = factor(paste("Condition Data:",condition))) %>%
  arrange(time)

ggplot(dflong,aes(x=condition_model,y=cor)) +
  facet_grid(condition~sid,
             labeller=labeller(sid=label_both)) +
  geom_point(alpha=0.5,position=position_jitter(width=0.05)) +
  stat_summary(aes(x=as.numeric(condition_model)+0.2)) +
  geom_hline(yintercept=0,linetype=2) +
  ylab('Correlation') + xlab('Condition Model')
ggsave(file.path(dir,'4_condition_models.pdf'))
