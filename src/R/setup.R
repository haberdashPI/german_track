base_dir = getwd()
library(ggplot2)
library(dplyr)
library(tidyr)
library(stringr)
library(rjson)

if(basename(base_dir) != 'german_track'){ ## TODO: change to 'german_track'
    warning(paste('Expected root directory to be named "eeg_atten". Was',
                  basename(base_dir)))
}
analysis_dir = file.path(base_dir,'scripts','R')
cache_dir = file.path(base_dir,'_research','cache','cache')
processed_datadir = file.path(base_dir,'data','processed')
raw_datadir = file.path(base_dir,'data','raw')
plot_dir = file.path(base_dir,'plots')
