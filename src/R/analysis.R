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
        across(all_of(cols), list(
            med = median,
            `05` = ~ posterior_interval(matrix(.x))[,1],
            `95` = ~ posterior_interval(matrix(.x))[,2],
            pd = ~ p_direction(.x)[[1]]
        )))
}

effect_table = function(df, digits = 3){
    limit = 10^(-digits)
    df %>%
        mutate(across(matches('_p$'), ~ ifelse(.x <= limit,
            str_c('≤',limit),
            str_c(' ',as.character(round(.x, digits = digits)))))) %>%
        knitr::kable(digits = digits)
}

pairwise = function(df, ..., bothdir = F){
    cols = names(select(df, ...))
    n = length(cols)
    if(bothdir){
        for(i in 1:n){
            for(j in 1:n){
                if(i != j){
                    df = cbind(df, newcol = df[,cols[i]] - df[,cols[j]])
                    names(df)[names(df) == "newcol"] = str_c(cols[i],' - ',cols[j])
                }
            }
        }
    }else{
        for(i in 1:(n-1)){
            for(j in (i+1):n){
                df = cbind(df, newcol = df[,cols[i]] - df[,cols[j]])
                names(df)[names(df) == "newcol"] = str_c(cols[i],' - ',cols[j])
            }
        }
    }
    df
}

effect_list = function(df, ...){
    inlist = function(val, vars, result = list()){
        if(length(vars) > 1){
            result[[first(vars)]] = inlist(val, vars[2:length(vars)], result[[first(vars)]])
            result
        }else{
            result[[first(vars[1])]] = as.list(val[[1]])
            result
        }
    }
    nested = df %>% nest_by(...)
    result = list()
    for(r in 1:nrow(nested)){
        # cat('row: ',r,'\n')
        result = inlist(nested[r,]$data, select(nested[r,], -data), result)
    }

    result
}

effect_json = function(df,  label, ...){
    result = list()
    result[[label]] = df %>%
        mutate(across(everything(), unname)) %>%
        effect_list(...)
    result %>% toJSON %>% cat(file = file.path(stat_dir, str_c(label, '.json')))
}
