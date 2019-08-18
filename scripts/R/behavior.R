# look at the behaviora data of the 8 EEG participants
source("src/R/setup.R")

df = NULL
for(file in list.files(data_dir,"sound_events.*csv")){
    df_ = read.csv(file.path(data_dir,file))
    df = rbind(df,df_)
}
dprime = function(hits,falarm){
    qnorm(hits) - qnorm(falarm)
}

summary = df %>% group_by(sid,condition) %>%
    summarize(dp = dprime())
ggplot(df,aes(x = sid, ))
