source("src/R/setup.R")
library(magrittr)
library(cowplot)
library(forcats)

# NOTE: sid 15 has missing trials in the eeg and there is not an obvious
# way to infer which events are missing (it appears some triggers did not
# record at it isn't clear which ones from looking at the data)

# missing SID:
# 1-7: piloting data using different stimulus
# 15, 20, 23: badly recorded event file, can't be certain which events correspond to what stimuli
#  (might be able to fix some of these)
# 26: badly reorded event file, and listener reported memorizing stimuli to determine the correct response (!!)

for(sid in c(8:14,16:19, 21:22, 24:25, 27:35)){

cat(paste0('Sid: ',sid,'\n'))
efraw = read.csv(file.path(processed_datadir,'eeg',sprintf("eeg_events_%03d.csv",sid)))
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


presfiles = list.files(file.path(raw_datadir,'eeg'),sprintf("%04d.*log",sid))
if(length(presfiles) > 1){
    msg = sprintf("Multiple files matching pattern for sid = %d:",sid)
    stop(do.call(paste,c(list(msg), as.list(presfiles),list(sep="\n"))))
}
presfile = file.path(raw_datadir,'eeg',presfiles)

pf = read_experiment_events(presfile, response_codes = c(2,3))

# check the counts of each condition (should be 50 for each)
pf %>% group_by(condition) %>% summarize(count = length(sound_index))

# browser()

if(sid == 9){
    # subject 9's first trial was not recorded (in the eeg)
    sound_events = filter(ef,bit == 4)
    pf = pf[2:nrow(pf),]
    pf$trial = pf$trial - 1
    if(nrow(sound_events) != 153)
        stop(sprintf("Unexpected number of rows: %d",nrow(sound_events)))

    pf = pf %>%
        rename(pres_time = time) %>%
        mutate(sample = sound_events$sample[trial],
               time = sound_events$time[trial]) %>%
        mutate(trial = trial - ifelse(trial <= 49,0,ifelse(trial <= 103,2,4)))
}else if(sid == 23){
    # presentation failed to record the final trial
}else{
    sound_events = filter(ef,bit == 5)
    if(nrow(sound_events) != 154)
        stop(sprintf("Unexpected number of rows: %d",nrow(sound_events)))

    pf = pf %>%
        rename(pres_time = time) %>%
        mutate(sample = sound_events$sample[trial],
               time = sound_events$time[trial]) %>%
        mutate(trial = trial - ifelse(trial <= 50,0,ifelse(trial <= 104,2,4))) %>%
        arrange(trial)
}

# if we have indexing errors above, check them using the below plot
# plotdf = ef %>% filter(bit == 5) %>% mutate(index = 1:length(bit))
# ggplot(plotdf,aes(x=time,y=index)) + geom_point() + geom_text(aes(label=index),nudge_x = 50)

if(any(diff(pf$trial) != 1)){
    stop("Improper trial indices!")
}


pf %>%
    select(trial,sample,time,condition,reported_target,sound_index) %>%
    write.csv(file.path(processed_datadir,'eeg',
        sprintf("sound_events_%03d.csv",sid)),
        row.names=FALSE)

}
