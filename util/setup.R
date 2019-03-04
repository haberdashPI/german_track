base_dir = getwd()
library(ggplot2)
library(dplyr)
library(tidyr)
library(stringr)

if(basename(base_dir) != 'german_track'){ ## TODO: change to 'german_track'
  warning(paste('Expected root directory to be named "eeg_atten". Was',
                basename(base_dir)))
}

analysis_dir = file.path(base_dir,'analyses')
model_dir = file.path(base_dir,'model')
cache_dir = file.path(analysis_dir,'cache')
data_dir = file.path(base_dir,'data')
plot_dir = file.path(base_dir,'plots')

if(Sys.info()["nodename"] == 'Mycroft.local'){
  raw_data_dir = '/Volumes/MiguelJr/Research/Experiments/trackgerman/data/'
}else{
  raw_data_dir = '/Volumes/Data/Little_german_track_2019-02-27/'
}
