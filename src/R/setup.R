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
dates = fromJSON(file = file.path(base_dir,'dateconfig.json'))

analysis_dir = file.path(base_dir,'scripts','R')
cache_dir = file.path(base_dir,'_research','cache','cache')
data_dir = file.path(base_dir,'data','exp_pro','eeg',dates$data_dir)
raw_data_dir = file.path(base_dir,'data','exp_raw','eeg')
plot_dir = file.path(base_dir,'plots')
