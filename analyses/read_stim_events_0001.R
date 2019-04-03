library(magrittr)
library(dplyr)
library(ggplot2)
library(cowplot)

source('util/setup.R')

## 0001 data was collected before some improvements to the trigger
## setup, so it's preprocessing is a special case, handled in this file

efraw = read.csv(file.path(data_dir,'eeg_events_0001.csv'))
ef = NULL
sr = 2048
for(bit in 0:7){
    ef = rbind(ef,efraw %>%
               filter(bitwAnd(code,2^bit) > 0) %>%
               mutate(bit = bit) %>%
               select(bit,sample))
}
ef = ef %>% arrange(sample) %>%
    mutate(time = sample/sr)
p1 = ggplot(ef,aes(x=time/60,y=bit,color=factor(bit))) + geom_point() +
    xlab('minutes')

presfile = file.path(raw_data_dir,'2018_01-24-0001_DavidLittle_presentation.log')

raw_pf = read.table(presfile,header=T,skip=3,sep='\t',blank.lines.skip=T,fill=T)
## names(pf) %<>% tolower # make all columns lower case

## retinterpret to make each trial a single row
pf = raw_pf %>% rename(subtrial=Trial) %>%
    mutate(trial = cumsum(Event.Type=='Sound')) %>%
    group_by(trial) %>%
    summarize(response =
              last(Code[Event.Type == 'Response' & Code %in% c(2,3)]),
          time = first(Time) / 10^4, # in seconds
          condition = first(block_type.str.),
          trial_block_offset = first(trial_order.num.),
          sound_index = first(trial_file.num.),
          response_time =
              last(TTime[Event.Type == 'Response' & Code %in% c(2,3)]) / 10^4)

    ## drop practice trials
    pf = pf %>% filter(condition %in% c('test','object','feature')) %>%
        arrange(time)
    ## align times
    last_time = last(ef$time)
    pf = pf %>% mutate(time = time + (last_time - last(time)))

    p2 = ggplot(pf,aes(x=time/60,y=response,color=response)) + geom_point() +
        xlab('mintues')

    ## compare events
    plot_grid(p1,p2,nrow=2,ncol=1,align='v')

    ####################
    ## compare responses
    presr = pf %>% select(time,response) %>%
        mutate(response = ifelse(response == 2,'yes','no'))
    presr$src = 'presentation'

    eegr = ef %>%
        filter(bit %in% c(0,1)) %>%
        group_by(time) %>%
        summarize(response = ifelse(any(bit == 0),'no','yes'))
    eegr$src = 'eeg'

    responses = rbind(presr,eegr)
    ggplot(responses,aes(x=time,y=src,color=response)) + geom_point()

    ## inject sound codes
    sound_events = filter(ef,bit == 4)
    sound_events$condition = 'unknown'
    sound_events$sound_index = NA
    sound_events$pres_trial = NA
    sound_events$pres_time = NA
    for(pres_row in 1:nrow(pf)){
        eeg_row = last(which(sound_events$time <
                             as.numeric(pf[pres_row,'time']) - 6))
        sound_events[eeg_row,'condition'] = as.character(pf[pres_row,]$condition)
        sound_events[eeg_row,'sound_index'] = pf[pres_row,'sound_index']
        sound_events[eeg_row,'pres_trial'] = pf[pres_row,'trial']
        sound_events[eeg_row,'pres_time'] = pf[pres_row,'time']
        sound_events[eeg_row,'response'] = pf[pres_row,'response']
    }

    # verify trial alignment
    ggplot(sound_events,aes(x=time,
                            y=pres_trial,
                            color=condition)) +
geom_point() + geom_line()

# verify stimulus distribution
ggplot(sound_events,aes(x=time,
                        y=ifelse(is.na(sound_index),75,sound_index),
                        color=condition)) +
geom_point()

## examine time differences
ggplot(sound_events,aes(x=time,y=pres_time)) +
    geom_point() + geom_abline(intercept=0,slope=1)
ggplot(sound_events,aes(x=(time-pres_time) - mean(time-pres_time))) +
    geom_histogram()
ggplot(sound_events,aes(x=(time-pres_time) - mean(time-pres_time))) +
    geom_density()
ggplot(sound_events,aes(x=(time-pres_time) - mean(time-pres_time),
                        y=rnorm(length(time)))) + geom_point(shape='o',size=3)
sd(sound_events$time - sound_events$pres_time)

## save sound events
sound_events %>%
    select(sample,time,condition,response,sound_index) %>%
    write.csv(file.path(data_dir,'sound_events_0001.csv'))
