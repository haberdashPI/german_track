library(magrittr)
library(dplyr)
library(ggplot2)
library(cowplot)

source("util/setup.R")

sid = 10

efraw = read.csv(file.path(data_dir,sprintf("eeg_events_%03d.csv",sid)))
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
    xlab("minutes")
p1

presfiles = list.files(file.path(raw_data_dir),sprintf("%04d.*log",sid))
if(length(presfiles) > 1){
    stop(do.call(paste,c(list(sprintf("Multiple files matching pattern for sid = %d:",sid)),
                         as.list(presfiles),list(sep="\n"))))
}
presfile = file.path(raw_data_dir,presfiles)

raw_pf = read.table(presfile,header=T,skip=3,sep="\t",blank.lines.skip=T,fill=T)

pf = raw_pf %>% rename(subtrial=Trial) %>%
    mutate(trial = cumsum(Event.Type=="Sound")) %>%
    group_by(trial) %>%
    summarize(response =
              last(Code[Event.Type == "Response" & Code %in% c(2,3)]),
          time = first(Time) / 10^4, # in seconds
          condition = first(block_type.str.),
          trial_block_offset = first(trial_order.num.),
          sound_index = first(trial_file.num.),
          response_time =
              last(TTime[Event.Type == "Response" & Code %in% c(2,3)]) / 10^4)

    pf = pf %>% filter(condition %in% c("test","object","feature"),
                       !is.na(response)) %>%
    arrange(time)

# check the counts of each condition (should be 50 for each)
pf %>% group_by(condition) %>% summarize(count = length(sound_index))

sound_events = filter(ef,bit == 5)
# sound_events = filter(ef,bit == 4) # for subject 9 only

# note: comment out the below for subject 9
if(nrow(sound_events) != 154)
    stop(sprintf("Unexpected number of rows: %d",nrow(sound_events)))

# these extra rows, which we're skipping, are practice trials during the intro
# to the 'feature' and 'object' conditions of the experiment

# duriung recording for subject 9, the first event was lost
sound_events = sound_events[c(1:50,53:102,105:154),]
# sound_events = sound_events[c(2:50,53:102,105:154)-1,] # subject 9 only
# pf = pf[2:nrow(pf),] # subject 9 only

pf = pf %>%
    rename(pres_time = time) %>%
    mutate(sample = sound_events$sample,
           time = sound_events$time)

pf %>%
    select(sample,time,condition,response,sound_index) %>%
    write.csv(file.path(data_dir,sprintf("sound_events_%03d.csv",sid)))

pf %>%
    mutate(index = as.numeric(condition)) %>%
    select(time,index,sound_index) %>%
    write.table(file.path(data_dir,sprintf("sound_events_%03d.txt",sid)),
                quote=F,sep="\t",row.names=F)
