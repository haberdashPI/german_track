library(dplyr)
library(ggplot2)
library(tidyr)

dir = paste('plots',Sys.Date(),sep='_')
dir.exists(dir) || dir.create(dir)

## TODO:
## along x-axis plot all correctly identified targets next to one another (in
## order) along the second part of the x-axis plot all incorrectly identified
## targets on the y-axis is the relative correlation score of the target
## voice within some window around the target time.

df = read.csv('target_correlations.csv')
df = df %>%
  rename(hitcor = hit_target_cor,
         misscor = miss_target_cor,
         globalcor = test_condition_cor,
         objectcor = object_condition_cor,
         featurecor = feature_condition_cor) %>%
  mutate(condition = recode(condition,test = 'globalc'),
         target_heard = response == 2,
         target_present = !is.nan(target_time),
         hit = target_heard & target_present,
         miss = !target_heard & target_present) %>%
  arrange(time)

ggplot(subset(df,target_present & !is.na(hit) & !is.na(miss)),
       aes(x=hit,y=hitcor - misscor)) +
  facet_grid(condition~.) +
  geom_point(position=position_jitter(width=0.05)) +
  stat_summary(aes(x=as.numeric(hit)+1.1)) +
  geom_hline(yintercept=0,linetype=2)
ggsave(file.path(dir,'1_hit_vs_miss_diff.pdf'))

ggplot(subset(df,target_present & !is.na(hit) & !is.na(miss)),
       aes(x=misscor,y=hitcor)) +
  facet_grid(condition~hit) +
  geom_point(position=position_jitter(width=0.05)) +
  geom_hline(yintercept=0,linetype=2) +
  geom_abline(intercept=0,slope=1)
ggsave(file.path(dir,'2_hit_vs_miss_diag.pdf'))

t.test(I(misscor-hitcor) ~ hit,
       subset(df,target_present & !is.na(hit) & condition == 'object'))
t.test(hitcor ~ hit,
       subset(df,target_present & !is.na(hit) & condition == 'object'))
t.test(hitcor ~ hit,subset(df,target_present & !is.na(hit)))
t.test(I(misscor-hitcor) ~ hit,subset(df,target_present & !is.na(hit)))

dfcor = df %>%
  gather(condition_cor,cor,objectcor,featurecor,globalcor) %>%
  mutate(condition_cor = factor(condition_cor)) %>%
  arrange(time)


ggplot(dfcor,aes(x=recode(condition_cor,
                          globalcor = 'Mixture (global)',
                          featurecor = 'Right Channel (feature)',
                          objectcor = 'Male Speaker (object)'),
                 y=cor)) +
  xlab('Envelope') + ylab('Correlation') +
  geom_point(position=position_jitter(width=0.05)) +
  stat_summary(aes(x=as.numeric(condition_cor)+0.15)) +
  geom_hline(yintercept=0,linetype=2) +
  facet_grid(condition~.,
             labeller=as_labeller(c(globalc = 'Condition - Global',
                                    feature = 'Condition - Feature',
                                    object = 'Condition - Object')))
ggsave(file.path(dir,'3_condition_correlation.pdf'))
