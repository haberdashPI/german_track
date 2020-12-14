source("src/R/setup.R")

# moves the json formatted stats results to the git repository for paper1

data = NULL
for(file in list.files(stat_dir)){
    data = c(data, fromJSON(paste(readLines(file.path(stat_dir, file)), collapse="")))
}

data %>% toJSON %>%
    cat(file = file.path(base_dir, 'publications', 'paper1', 'stats.json'))
