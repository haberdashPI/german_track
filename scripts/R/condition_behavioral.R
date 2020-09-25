source("src/R/setup.R")
library(dplyr)
library(ggplot2)
library(stringr)

presfiles = list.files(file.path(raw_datadir,'behavioral'),sprintf("*.*log",sid))

for(filename in presfiles){
    cat(paste0('File: ',filename,'\n'))
    sid = as.numeric(str_extract(filename, '^[0-9]+'))
    presfile = file.path(raw_datadir, 'behavioral', filename)
    pf = read_experiment_events(presfile)

    # check the counts of each condition (should be 50 for each)
    pf %>% group_by(condition) %>% summarize(count = length(sound_index))

    dir.create(file.path(processed_datadir,'behavioral'), recursive = TRUE)
    pf %>%
        select(trial,time,condition,reported_target,sound_index) %>%
        write.csv(file.path(processed_datadir,'behavioral',
            sprintf("sound_events_%03d.csv",sid)),
            row.names=FALSE)
}

