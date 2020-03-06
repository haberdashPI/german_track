run(fullfile('..','..','src','matlab','util','setup.m'));
usecache = 0;

eegfiles = dir(fullfile(raw_data_dir,'*.bdf'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot configuration

% trial plot configuration
plot_cfg = [];
plot_cfg.viewmode = 'vertical';
plot_cfg.eegscale = 1;
plot_cfg.ylim = [-20 20];
plot_cfg.preproc.detrend = 'yes';
plot_cfg.preproc.demean = 'no';
plot_cfg.blocksize = 8;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% find the length of all stimuli

soundsdir = fullfile(stim_data_dir,'mixtures','testing');
sounds = sort({dir(fullfile(soundsdir,'*.wav')).name});
sound_lengths = zeros(length(sounds),1);
set i = 1:length(sounds)
    [x,fs] = audioread(char(fullfile(soundsdir,sounds(i))));
    sound_lengths(i) = size(x,1) / fs;
end

% STEPS to get working:
% 1. do on a per trial basis, rather than across all trials
%    (the trials are quite long, and there's data outside the trials
%     that we don't need to clean)
% 2. get outlier detection working (see if it is worth it)
% 3. find and remove eye blinks ussing DSS

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subject 009

% get filename and SID
file = fullfile(raw_data_dir,eegfiles(2).name);
numstr = regexp(file,'([0-9]+)_','tokens');
sid = str2num(numstr{1}{1});

% read in the events
event_file = fullfile(data_dir,sprintf('sound_events_%03d.csv',sid));
stim_events = readtable(event_file);

% read in eeg data header
head = ft_read_header(file);
% for SID 9, we recorded some extra channels (skip them)
head.label = head.label([1:64 129:134]);
head.nChans = 70;

% define trial boundaries
cfg = [];
cfg.dataset = file;
trial_lengths = round(sound_lengths(stim_events.sound_index)*head.Fs);
cfg.trl = [stim_events.sample stim_events.sample + trial_lengths zeros(length(stim_events.sample),1)];
cfg.continuous = 'no';
cfg.channel = [1:64 129:134]; % for SID 9, extra channels were recorded
eeg = ft_preprocessing(cfg);
raw_eeg = eeg;
ntrials = length(eeg.trial);

% downsample and demean the data
eeg = gt_settrials(@nt_dsample,eeg,8,'progress','resampling...');
eeg = gt_settrials(@nt_demean,eeg);
head.Fs = head.Fs / 8;
eeg.time = gt_fortrials(@(data) ((1:size(data,1))/head.Fs),eeg)';
eeg.fsample = eeg.fsample / 8;
eeg.sampleinfo = round([stim_events.time*eeg.fsample stim_events.time*eeg.fsample + (cellfun(@(x) size(x,2),eeg.trial))' - 1]);

% find bad channels
freq = 0.5;
all_bad_indices = gt_fortrials(@nt_find_bad_channels,eeg,freq,3,[],3,'channels',1:64);
badfreqs = tabulate(horzcat(all_bad_indices{:}));
bad_indices = badfreqs(badfreqs(:,2)/ntrials > freq,1);
head.label(bad_indices) % run this line to see which indices are bad

% interpolate bad channels
coords = nt_proximity('biosemi64.lay',63);
eeg = gt_settrials(@gt_interpolate_bad_channels,eeg,bad_indices,coords,'channels',1:64);

% visualize the data
ft_databrowser(plot_cfg, eeg);

% polynomial detrending
[trials,w] = gt_fortrials(@gt_detrend,eeg,[1 5],'progress','detrending...');
eeg.trials = cellfun(@(x) x',trials,'UniformOutput',false);
this_plot = plot_cfg;
this_plot.preproc.detrend = 'no';
ft_databrowser(this_plot, eeg);

% eyeblin removal

% step 1: find regions of likely eye blinks and movement
eog = 67:70; % the sensors near the eyes
eyes = gt_fortrials(@(x) x(:,eog),eeg);
eyes = vertcat(eyes{:});
[B,A]=butter(2,1/(head.Fs/2), 'high');
tmp=nt_pca(filter(B,A,eyes));
mask=abs(tmp(:,1))>3*median(abs(tmp(:,1)));
plot([eyes mask*200])

% step 2: find components using
C0=nt_cov(x);
C1=nt_cov(bsxfun(@times, x,mask));

% do we find any new bad indices? no, the found index doesn't look bad
% bad_indices = nt_find_bad_channels(eeg(:,1:64),0.5,[],[],5);
