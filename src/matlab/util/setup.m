function setup

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

fprintf('Base directory is %s\n',base_dir);
addpath(fullfile(base_dir,'src','matlab','util'));

% fieldtrip can be installed from http://www.fieldtriptoolbox.org/
% add it to the MATLAB path
ft_defaults;
% download https://github.com/sofacoustics/API_MO and add to MATLAB path
SOFAstart;

global analysis_dir;
global cache_dir;
global data_dir;
global raw_data_dir;
global stimulus_dir;

analysis_dir = fullfile(base_dir,'scripts','matlab');
cache_dir = fullfile(base_dir,'_research','cache');
data_dir = fullfile(base_dir,'data','exp_pro','eeg');
stimulus_dir = fullfile(base_dir,'src','matlab','stimuli');

[ret,hostname] = system('hostname');
config = read_json(fullfile(base_dir,'data','exp_raw','config.json'));

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

end
