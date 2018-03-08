base_dir = getwd()

if(basename(base_dir) != 'eeg_atten'){ ## TODO: change to 'german_track'
  warning(paste('Expected root directory to be named "eeg_atten". Was',
                basename(base_dir)))
}

analysis_dir = file.path(base_dir,'analysis')
model_dir = file.path(base_dir,'model')
cache_dir = file.path(analysis_dir,'preprocessed_data')
data_dir = file.path(base_dir,'data')

if(Sys.info()["nodename"] == 'Claude.local'){
  raw_data_dir = '/Volumes/MiguelJr/Research/Experiments/trackgerman/data/'
}else{
  raw_data_dir = '/Users/davidlittle/Data/EEGAttn_David_Little_2018_01_24/'
}
