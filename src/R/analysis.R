
read_experiment_events = function(presfile, response_codes = c('y','n')){
    raw_pf = read.table(presfile,header=TRUE,skip=3,sep="\t",
        blank.lines.skip=TRUE,fill=TRUE)

    pf = raw_pf %>% rename(subtrial=Trial) %>%
        mutate(trial = cumsum(Event.Type=="Sound")) %>%
        mutate(block_type.str. = na_if(block_type.str., "")) %>%
        group_by(trial) %>%
        summarize(response =
                last(Code[Event.Type == "Response" & Code %in% response_codes]),
            time = first(Time) / 10^4, # in seconds
            condition = first(block_type.str.),
            trial_block_offset = first(trial_order.num.),
            sound_index = first(trial_file.num.),
            response_time =
                last(TTime[Event.Type == "Response" & Code %in% response_codes]) / 10^4)
    pf$reported_target = pf$response == response_codes[1]

    pf = pf %>% filter(condition %in% c("test","object","feature"),
                    !is.na(response)) %>%
        arrange(time) %>%
        mutate(trial = trial - first(trial)+1)

    pf$condition = fct_recode(pf$condition,
        "global" = "test",
        "spatial" = "feature",
        "object" = "object"
    )

    return(pf)
}
