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

fprintf('Base directory is %s\n',base_dir);
for listing = dir(fullfile(base_dir,'src','matlab','external'))
    if listing.isdir
        addpath(listing.name)
    end
end
addpath(fullfile(base_dir,'src','matlab','util'));

% fieldtrip can be installed from http://www.fieldtriptoolbox.org/
% add it to the MATLAB path
ft_defaults;
% download https://github.com/sofacoustics/API_MO and add to MATLAB path
SOFAstart;

analysis_dir = fullfile(base_dir,'scripts','matlab');
cache_dir = fullfile(base_dir,'_research','cache');
processed_datadir = fullfile(base_dir,'data','processed');
raw_stim_dir = fullfile(base_dir,'data','raw','stimuli');
stim_datadir = fullfile(base_dir,'data','processed','stimuli');
stimulus_dir = fullfile(base_dir,'src','matlab','stimuli');
raw_datadir = fullfile(base_dir,'data','raw','eeg');
