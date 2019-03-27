% NOTE: matlab should be called
base_dir = pwd();
[base_dir_rest,name] = fileparts(base_dir);
if strcmp(name,'util')
  base_dir = base_dir_rest;
  cd(base_dir);
end

[~,name] = fileparts(base_dir);
if ~strcmp(name,'german_track')
  warning(['Expected root directory to be named "german_track". Was ' base_dir]);
end

addpath('analyses');
addpath('util');

% fieldtrip can be installed from http://www.fieldtriptoolbox.org/
% add it to the MATLAB path
ft_defaults;
% download https://github.com/sofacoustics/API_MO and add to MATLAB path
SOFAstart;

global analysis_dir;
global cache_dir;
global model_dir;
global data_dir;
global raw_data_dir;
global stimulus_dir;

analysis_dir = fullfile(base_dir,'analyses');
cache_dir = fullfile(base_dir,'analyses','cache');
model_dir = fullfile(analysis_dir,'models');
data_dir = fullfile(base_dir,'data');
stimulus_dir = fullfile(base_dir,'stimuli');

[ret,hostname] = system('hostname');
config = read_json(fullfile(base_dir,'config.json'));

match = 0;
default_i = 1;
for i = 1:length(config)
    if strcmp(config(i).host,'default')
        default_i = i;
    elseif startsWith(hostname,config(i).host)
        raw_data_dir = config(i).raw_data_dir;
        match = 1;
    end
end
if ~match
  raw_data_dir = config(default_i).raw_data_dir;
end
