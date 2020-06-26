library(dplyr)
library(ggplot2)
library(tidyr)

source('util/setup.R')

dir = file.path(plot_dir,paste('run',Sys.Date(),sep='_'))
dir.exists(dir) || dir.create(dir)

df = NULL
cor_files = list.files(cache_dir,pattern='target_2.0seconds.*_cor_data.csv')
for(file in cor_files){
  dff = read.csv(file.path(cache_dir,file))
  df = rbind(df,dff)
}

dfstim = NULL
for(sid in unique(df$sid)){
  dff = read.csv(file.path(processed_datadir,'eeg',sprintf('sound_events_%03d.csv',sid)));
  dfstim = rbind(dfstim,dff)
}

df$condition = dfstim[df$trial,]$condition
fs = 64

dfstand = df %>%
  arrange(sid,model,trial) %>%
  group_by(sid,model,trial) %>%
  filter(!is.nan(prediction),!is.nan(response)) %>%
  mutate(sample = row_number(trial),
         time = sample/fs,
         response = response / sd(response),
         prediction = prediction / sd(prediction),
         cor = cor(response[response != 0],prediction[response != 0]))

for(sid_ in unique(dfstand$sid)){
  for(condition_ in unique(dfstand$condition)){
    dftest = dfstand %>%
      gather(kind,level,response,prediction) %>%
      filter(sid == sid_,condition == condition_,model == 'hit_target',!is.na(cor))

    ggplot(subset(dftest,sid == sid_),aes(x=time,y=level,color=kind)) +
      geom_line(alpha=0.5) +
      scale_color_brewer(palette='Set1') +
      facet_wrap(~paste("Trial",sprintf("%03d",trial),"(c =",round(cor,3),")")) +
      theme_classic()
    ggsave(file.path(dir,
                     sprintf('individual_sid_%03d_condition_%s.pdf',sid_,condition_)))

    dftest = dfstand %>%
      filter(sid == sid_,condition == condition_,model == 'hit_target',!is.na(cor))

    ggplot(dftest,aes(x=response,y=prediction)) +
      geom_point(alpha=0.2) +
      facet_wrap(~paste("Trial",sprintf("%02d",trial),"(c =",round(cor,3),")")) +
      theme_classic()
    ggsave(file.path(dir,
                     sprintf('individual_cor_sid_%03d_condition_%s.pdf',sid_,condition_)))

  }
}
