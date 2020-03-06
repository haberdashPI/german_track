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
% subject 009

% STEPS to get working:
% 1. do on a per trial basis, rather than across all trials
%    (the trials are quite long, and there's data outside the trials
%     that we don't need to clean)
% 2. get outlier detection working (see if it is worth it)
% 3. find and remove eye blinks ussing DSS

% get filename and SID
file = fullfile(raw_data_dir,eegfiles(2).name);
numstr = regexp(file,'([0-9]+)_','tokens');
sid = str2num(numstr{1}{1});

% load and downsample the data
head = ft_read_header(file);
head.label = head.label([1:64 129:134]);
eeg = ft_read_data(file)';
eeg = eeg(:,[1:64 129:134]);
eeg = nt_dsample(eeg,8);
eeg = nt_demean(eeg);
fs = head.Fs / 8; % 256 Hz
head.Fs = fs;

% read in the events
event_file = fullfile(data_dir,sprintf('sound_events_%03d.csv',sid));
stim_events = readtable(event_file);

% drop samples after the end of the last trial (to avoid wacky channels as I
% unplug the device)
last_sample = round((stim_events(end,:).time+10) * fs);
eeg = eeg(1:last_sample,:);

% find bad channels, interpolate
bad_indices = nt_find_bad_channels(eeg(:,1:64),0.5,[],[],100);
head.label(bad_indices) % run this line to see which indices are bad

% visualize the bad channels next to neighboring good channels
check_indices = [bad_indices; [bad_indices + 1]];
check_indices = unique(check_indices(:));
ft_databrowser(plot_cfg,gt_tofieldtrip(eeg,head,check_indices));

% select the actually bad indices
bad_indices = bad_indices(2);

coords = nt_proximity('biosemi64.lay',63);
[toGood,fromGood]=nt_interpolate_bad_channels(eeg(:,1:64),bad_indices,coords);
eeg(:,1:64)=eeg(:,1:64)*(toGood*fromGood);

% visualize the data (with some basic preprocessing to make the data viewable)
ft_databrowser(plot_cfg, gt_tofieldtrip(eeg,head));

% detrend: order of 1 point per 8 seconds
[eeg,w]=nt_detrend(eeg,100);
this_plot = plot_cfg;
this_plot.preproc.detrend = 'no';
ft_databrowser(this_plot, gt_tofieldtrip(eeg,head));

% detect channel specific glitches
[w,y]=nt_outliers(eeg,w,2,3);

% do we find any new bad indices? no, the found index doesn't look bad
% bad_indices = nt_find_bad_channels(eeg(:,1:64),0.5,[],[],5);
