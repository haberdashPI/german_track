base_dir = getwd()
if(basename(base_dir) != 'german_track'){
    warning(paste('Expected root directory to be named "german_track". Was',
                  basename(base_dir)))
}

library(ggplot2)
library(dplyr)
library(tidyr)
library(stringr)
library(rjson)
library(forcats)
library(rstanarm)
options(mc.cores = parallel::detectCores())

source("src/R/analysis.R")

analysis_dir = file.path(base_dir,'scripts','R')
cache_dir = file.path(base_dir,'_research','cache','cache')
processed_datadir = file.path(base_dir,'data','processed')
raw_datadir = file.path(base_dir,'data','raw')
plot_dir = file.path(base_dir,'plots')
