library(dplyr)
library(ggplot2)
library(tidyr)

dir = paste('plots',Sys.Date(),sep='_')
dir.exists(dir) || dir.create(dir)

df = read.csv('target_cor_data.csv')
dfstim = read.csv('sound_events.csv')

model_names = c('hit_target',
                'miss_target',
                'test_condition',
                'object_condition',
                'feature_condition')

df$condition = dfstim[df$trial,]$condition
df$model = model_names[df$model_index]
fs = 64

dfstand = df %>% group_by(model,trial) %>%
  mutate(sample = row_number(trial),
         time = sample/fs,
         response = response / sd(response),
         prediction = prediction / sd(prediction),
         cor = cor(response[response != 0],prediction[response != 0])) %>%
  arrange(model,trial)

dftest = dfstand %>%
  gather(kind,level,response,prediction) %>%
  filter(condition == 'test',model == 'test_condition')

ggplot(dftest,aes(x=time,y=level,color=kind)) +
  geom_line(alpha=0.5) +
  facet_wrap(~paste("Trial",sprintf("%02d",trial),"(c =",round(cor,3),")")) +
  theme_classic()

dftest = dfstand %>% filter(condition == 'test',model == 'test_condition',
                            response != 0)
ggplot(dftest,aes(x=response,y=prediction)) +
  geom_point(alpha=0.2) +
  facet_wrap(~paste("Trial",sprintf("%02d",trial),"(c =",round(cor,3),")")) +
  theme_classic()
