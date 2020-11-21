library(bayestestR)
library(purrr)

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

effect_summary = function(.df, ...){
    .df %>% transmute(...) %>% effect_summary_helper
}

effect_summary_helper = function(df){
    cols = names(df)
    cols = cols[!(cols %in% group_vars(df))]
    df = summarize(df,
        across(all_of(cols), mean, .names = '{.col}_mean'),
        across(all_of(cols), .names = '{.col}_05', ~ posterior_interval(matrix(.x))[,1]),
        across(all_of(cols), .names = '{.col}_95', ~ posterior_interval(matrix(.x))[,2]),
    )
}

pairwise = function(df){
    cols = names(df)
    n = length(cols)
    newdf = NULL
    for(i in 1:(n-1)){
        for(j in (i+1):n){
            if(length(newdf) == 0){
                newdf = data.frame(newcol = df[,cols[i]] - df[,cols[j]])
            }else{
                newdf = cbind(newdf, data.frame(newcol = df[,cols[i]] - df[,cols[j]]))
            }
            names(newdf)[names(newdf) == "newcol"] = str_c(cols[i],' - ',cols[j])
        }
    }
    newdf
}
