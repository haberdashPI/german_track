ft_defaults
analysisdir = 'models';
datadir = '/Users/davidlittle/Data/EEGAttn_David_Little_2018_01_24/';
eegfile = fullfile(datadir,'2018-01-24_0001_DavidLittle_biosemi.bdf');
modelfile_prefix = fullfile(analysisdir,...
                            '2018-01-24_0001_DavidLittle_targethit_');

stim_events = readtable('sound_events.csv');
head = ft_read_header(eegfile);
fs = head.Fs;

baseline = 0;
trial_len = 10;
baseline_samples = floor(baseline*fs);
trial_len_samples = floor(trial_len*fs);

% TODO: turn this into a script that reads files
% from the original directoy and stores it in the data file
% (so we normally can just manage the preprocessed data).
% preprocessing
eegfile_proc = [eegfile '.proc.mat'];

% define the trials
if exist(eegfile_proc,'file')
    error(['Cannot create a file with the same name as directory ' ...
           eegfile_proc]);
end

% load the trials
cfg = [];
cfg.dataset = eegfile;
cfg.trl = [stim_events{:,'sample'}-baseline_samples ...
           stim_events{:,'sample'}+trial_len_samples ...
           baseline_samples*ones(height(stim_events),1)];
cfg.continuous = 'yes';
cfg.channel = [1:128 257:262];

eeg_data = ft_preprocessing(cfg);

% downsample the trials
cfg = [];
cfg.resamplefs = 64;
eeg_data = ft_resampledata(cfg,eeg_data);

% re-reference the data
cfg = [];
cfg.refchannel = 'all';
cfg.reref = 'yes';
egg_data = ft_preprocessing(cfg,eeg_data);

% save to a file
save(eegfile_proc,'eeg_data');