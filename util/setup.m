% NOTE: matlab should be called
base_dir = pwd();
[base_dir_rest,name] = fileparts(base_dir);
if strcmp(name,'util')
  base_dir = base_dir_rest
  cd(base_dir);
end

[~,name] = fileparts(base_dir);
if ~strcmp(name,'eeg_atten') % TODO: change to german_track
  warning(['Expected root directory to be named "eeg_atten". Was ' base_dir]);
end

addpath('analyses');
addpath('util');
ft_defaults;

analysis_dir = fullfile(base_dir,'analyses');
cache_dir = fullfile(base_dir,'analyses','preprocessed_data');
model_dir = fullfile(analysis_dir,'models');
data_dir = fullfile(base_dir,'data');

[ret,name] = system('hostname');
if startsWith(name,'Claude.local')
  raw_data_dir = '/Volumes/MiguelJr/Research/Experiments/trackgerman/data/';
else
  raw_data_dir = '/Users/davidlittle/Data/EEGAttn_David_Little_2018_01_24/';
end
