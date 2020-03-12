% NOTE: matlab should be called
dirs = strsplit(pwd(),filesep);
while ~strcmp(dirs(end),'german_track')
    dirs = dirs(1:(end-1));
    if length(dirs) == 0
        error('Could not find base directory "german_track"');
    end
end
base_dir = join(dirs,filesep);
base_dir = base_dir{1};
cd(base_dir)

global base_dir;
global analysis_dir;
global cache_dir;
global data_dir;
global raw_data_dir;
global stim_data_dir;
global stimulus_dir;
global raw_stim_dir;

fprintf('Base directory is %s\n',base_dir);
addpath(fullfile(base_dir,'src','matlab','util'));

% fieldtrip can be installed from http://www.fieldtriptoolbox.org/
% add it to the MATLAB path
ft_defaults;
% download https://github.com/sofacoustics/API_MO and add to MATLAB path
SOFAstart;

dates = read_json(fullfile(base_dir,'dateconfig.json'));

analysis_dir = fullfile(base_dir,'scripts','matlab');
cache_dir = fullfile(base_dir,'_research','cache');
data_dir = fullfile(base_dir,'data','exp_pro','eeg',dates.data_dir);
raw_stim_dir = fullfile(base_dir,'data','exp_raw','stimuli');
stim_data_dir = fullfile(base_dir,'data','exp_pro','stimuli',dates.stim_data_dir);
stimulus_dir = fullfile(base_dir,'src','matlab','stimuli');
raw_data_dir = fullfile(base_dir,'data','exp_raw','eeg');
