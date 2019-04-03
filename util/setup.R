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

analysis_dir = file.path(base_dir,'analyses')
model_dir = file.path(base_dir,'model')
cache_dir = file.path(analysis_dir,'cache')
data_dir = file.path(base_dir,'data')
plot_dir = file.path(base_dir,'plots')

hostname = Sys.info()["nodename"]
match = F
default_i = 0
config = fromJSON(file = file.path(base_dir,'config.json'))
for(i in 1:length(config)){
    if(config[[i]]$host == 'default'){
        default_i = i
    }else if(startsWith(hostname,config[[i]]$host)){
        raw_data_dir = config[[i]]$raw_data_dir
        match = T
    }
}
if(!match){
    raw_data_dir = config[[default_i]]$raw_data_dir
}
