library(dplyr)
library(ggplot2)
library(tidyr)

source('util/setup.R')

dir = file.path(plot_dir,paste('run',Sys.Date(),sep='_'))
dir.exists(dir) || dir.create(dir)

df = NULL
cor_files = list.files(cache_dir,pattern='target_2.0seconds.*_cor.csv')
for(file in cor_files){
  dff = read.csv(file.path(cache_dir,file))

  dff = dff %>%
    rename(hitcor = hit_target_cor,
           misscor_nont = miss_nontarget_cor,
           misscor_mix = miss_mix_cor) %>%
    mutate(condition = recode(condition,test = 'globalc'),
           target_heard = response == 2,
           target_present = !is.na(target_time) & target_time > 0,
           hit = target_heard & target_present,
           miss = !target_heard & target_present) %>%
    arrange(time)

  df = rbind(df,dff)
}

bdf = df %>%
  mutate(correct = target_heard == target_present) %>%
  group_by(sid,condition,target_present) %>%
  summarize(correct = mean(correct,na.rm=T))

pos = position_jitter(width=0.1)
ggplot(bdf,aes(x=target_present,y=correct)) +
  geom_label(aes(label=sid),position=pos) +
  facet_grid(condition~.) + ylim(0.4,1) +
  geom_hline(yintercept=0.5,linetype=2) +
  ylab('Proportion Correct') + xlab('Target Present')
ggsave(file.path(dir,'1_behavioral_responses.pdf'))

ggplot(subset(df,target_present & !is.na(hit) & !is.na(miss)),
       aes(x=hit,y=hitcor - misscor_nont)) +
  facet_grid(condition~sid) +
  geom_point(alpha=0.5,position=position_jitter(width=0.05)) +
  stat_summary(aes(x=as.numeric(hit)+1.1)) +
  geom_hline(yintercept=0,linetype=2) +
  ylab('Target Correlation - Non-target Correlation') + xlab('Target Detected')

ggsave(file.path(dir,'2_target_vs_miss_non_targets.pdf'))

ggplot(subset(df,target_present & !is.na(hit) & !is.na(miss)),
       aes(x=hit,y=hitcor - misscor_mix)) +
  facet_grid(condition~sid) +
  geom_point(alpha=0.5,position=position_jitter(width=0.05)) +
  stat_summary(aes(x=as.numeric(hit)+1.1)) +
  geom_hline(yintercept=0,linetype=2) +
  ylab('Target Correlation - Mixture Correlation') + xlab('Target Detected')

ggsave(file.path(dir,'3_target_vs_miss_mix.pdf'))

## t.test(hitcor ~ hit,subset(df,target_present & !is.na(hit)))
## t.test(I(misscor-hitcor) ~ hit,subset(df,target_present & !is.na(hit)))

## t.test(hitcor ~ hit,
##        subset(df,target_present & !is.na(hit) & condition == 'globalc'))
## t.test(I(misscor-hitcor) ~ hit,
##        subset(df,target_present & !is.na(hit) & condition == 'globalc'))
