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
global processed_datadir;
global raw_datadir;
global stim_datadir;
global stimulus_dir;
global raw_stim_dir;
global plot_dir;

fprintf('Base directory is %s\n',base_dir);
addpath(fullfile(base_dir,'src','matlab','external','noisetools','NoiseTools'));
addpath(fullfile(base_dir,'src','matlab','external','nsltools'));
addpath(fullfile(base_dir,'src','matlab','external','sofa','API_MO'));
addpath(fullfile(base_dir,'src','matlab','util'));

% fieldtrip can be installed from http://www.fieldtriptoolbox.org/
% add it to the MATLAB path
ft_defaults;
SOFAstart;

analysis_dir = fullfile(base_dir,'scripts','matlab');
cache_dir = fullfile(base_dir,'_research','cache');
processed_datadir = fullfile(base_dir,'data','processed');
raw_stim_dir = fullfile(base_dir,'data','raw','stimuli');
stim_datadir = fullfile(base_dir,'data','processed','stimuli');
stimulus_dir = fullfile(base_dir,'src','matlab','stimuli');
raw_datadir = fullfile(base_dir,'data','raw','eeg');
plot_dir = fullfile(base_dir,'plots');
