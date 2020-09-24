source("src/R/setup.R")
library(dplyr)
library(ggplot2)

sid = 33
'~/Documents/work/projects/german_track/data/raw/behavioral/preeeg'
presfiles = list.files(file.path(raw_datadir,'behavioral','preeeg'),sprintf("%02d.*log",sid))
if(length(presfiles) > 1){
    msg = sprintf("Multiple files matching pattern for sid = %d:",sid)
    stop(do.call(paste,c(list(msg), as.list(presfiles),list(sep="\n"))))
}
presfile = file.path(raw_datadir,'behavioral','preeeg',presfiles)

# TODO: this is where I stopped; need to validate the read in data, make sure it makes sense
# and then figure out if anything about the subsequent processing needs to be changed
raw_pf = read.table(presfile,header=TRUE,skip=3,sep="\t",
    blank.lines.skip=TRUE,fill=TRUE)

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
    arrange(time) %>%
    mutate(trial = trial - first(trial)+1)

pf$condition = fct_recode(pf$condition,
    "global" = "test",
    "spatial" = "feature",
    "object" = "object"
)

# check the counts of each condition (should be 50 for each)
pf %>% group_by(condition) %>% summarize(count = length(sound_index))


