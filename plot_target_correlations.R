library(dplyr)
library(ggplot2)
library(tidyr)
## TODO:
## along x-axis plot all correctly identified targets next to one another (in
## order) along the second part of the x-axis plot all incorrectly identified
## targets on the y-axis is the relative correlation score of the target
## voice within some window around the target time.

df = read.csv('target_correlations.csv')
df = df %>%
  mutate(target_heard = response == 2) %>%
  gather(signal,cor,fem_young,fem_old,male) %>%
  arrange(time)

dfcor = df %>%
  group_by(sample,time,sound_index) %>%
  summarize(target_time = first(target_time),
            target_heard = first(target_heard),
            condition = first(condition),
            target_cor = ifelse(any(signal == target),cor[signal == target],NA),
            nontarget_cor = max(cor[signal != target] - target_cor)) %>%
  mutate(has_target = !is.na(target_time),
         hit = target_heard & has_target,
         miss = !target_heard & has_target)

ggplot(subset(dfcor,has_target & !is.na(hit)),
       aes(y=target_cor,x=nontarget_cor)) + geom_point() +
  facet_grid(~hit) + geom_abline(intercept=0,slope=1)

ggplot(subset(dfcor,has_target & !is.na(hit)),
       aes(y=target_cor,x=nontarget_cor)) + geom_point() +
  facet_grid(condition~hit) + geom_abline(intercept=0,slope=1)

ggplot(subset(dfcor,has_target & !is.na(hit)),
       aes(x=hit,y=target_cor-nontarget_cor)) +
  geom_point(position=position_jitter(width=0.05)) +
  stat_summary(aes(x=as.numeric(hit) + 1.1)) +
  geom_hline(yintercept=0,linetype=2) + xlab('')

ggplot(filter(dfcor,has_target & !is.na(hit)),aes(x=hit,y=target_cor)) +
  geom_point(position=position_jitter(width=0.05)) +
  stat_summary(aes(x=as.numeric(hit) + 1.1)) +
  geom_hline(yintercept=0,linetype=2)

ggplot(filter(dfcor,has_target & !is.na(hit)),aes(x=hit,y=target_cor)) +
  geom_point(position=position_jitter(width=0.05)) +
  stat_summary(aes(x=as.numeric(hit) + 1.1)) +
  geom_hline(yintercept=0,linetype=2) +
  facet_grid(condition~.)

t.test(target_cor ~ hit,subset(dfcor,!is.na(hit) & has_target))
t.test(with(subset(dfcor,has_target & !is.na(hit)),(target_cor > 0) == hit),mu=0.5)
t.test(with(subset(dfcor,has_target & !is.na(hit)),
       (target_cor-nontarget_cor > 0) == hit),mu=0.5)
