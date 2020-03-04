run(fullfile('..','..','src','matlab','util','setup.m'));
usecache = 1;

eegfiles = dir(fullfile(raw_data_dir,'*.bdf'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subject 008

% get filename and SID
file = fullfile(raw_data_dir,eegfiles(1).name);
numstr = regexp(file,'([0-9]+)_','tokens');
sid = str2num(numstr{1}{1});

% load and downsample the data
head = ft_read_header(file);
eeg = ft_read_data(file)';
eeg = nt_dsample(eeg,8);
fs = head.Fs / 8; % 256 Hz

% read in the events
event_file = fullfile(data_dir,sprintf('sound_events_%03d.csv',sid));
stim_events = readtable(event_file);

% find bad channels, interpolate
proportion=0.5; % criterion proportion of bad samples
thresh1=3; % threshold in units of median absolute value over all data
thresh2=[]; % absolute threshold
thresh3=[]; % absolute threshold applied to sns-processed data
bad_indices = nt_find_bad_channels(eeg(:,1:64),proportion,thresh1,thresh2,thresh3);
[toGood,fromGood]=nt_interpolate_bad_channels(x(:,1:128),iBad,'biosemi64.lay');
x(:,1:128)=x(:,1:128)*(toGood*fromGood);
% iBad=nt_find_bad_channels(x(:,129:end),proportion,thresh1,thresh2,thresh3);
% x(:,128+iBad)=0; % positions unknown, just zap

% for i = 1:length(eegfiles)
%     eegfile = eegfiles(i).name;
%     numstr = regexp(eegfile,'([0-9]+)_','tokens');
%     sid = str2num(numstr{1}{1});
%     result_file = fullfile(data_dir,sprintf('eeg_response_%03d.mat',sid));

%     if exist(result_file,'file') && usecache
%         warning(['The file ' result_file ' already exists. Skipping...']);
%         continue;
%     end

%     disp(['reading responses for ' eegfile]);
%     disp(['Found SID = ' num2str(sid)]);

%     event_file = fullfile(data_dir,sprintf('sound_events_%03d.csv',sid));
%     stim_events = readtable(event_file);
%     head = ft_read_header(fullfile(raw_data_dir,eegfile));
%     fs = head.Fs;

%     baseline = 0;
%     % TODO: define the length based on the audio
%     % to simplify my life later on
%     baseline_samples = floor(baseline*fs);
%     trial_len_samples = arrayfun(@(i)trial_length(fs,stim_events,i,0.5), 1:size(stim_events,1));

%     % load the trials
%     cfg = [];
%     cfg.dataset = fullfile(raw_data_dir,eegfile);
%     cfg.trl = [max(0,stim_events{:,'sample'}-baseline_samples) ...
%         min(head.nSamples,stim_events{:,'sample'}+trial_len_samples) ...
%         baseline_samples*ones(height(stim_events),1)];
%     cfg.continuous = 'yes';
%     if sid == 1
%         cfg.channel = [1:128 257:264];
%     elseif sid == 9
%         cfg.channel = [1:64 129:134];
%     else
%         cfg.channel = 1:70;
%     end

%     % Note: based on https://www.biorxiv.org/content/10.1101/530220v1
%     % we don't apply a high-pass filter: instead, during clean_data.m
%     % we use NoiseTools robust de-trending.

%     raw_eeg_data = ft_preprocessing(cfg);

%     % downsample the trials to 256 (minimal framerate to find artifacts)
%     cfg = [];
%     cfg.resamplefs = 256;
%     eeg_data = ft_resampledata(cfg,raw_eeg_data);

%     % re-reference the data
%     cfg = [];
%     cfg.refchannel = 'all';
%     cfg.reref = 'yes';
%     eeg_data = ft_preprocessing(cfg,eeg_data);

%     % save to a file
%     ft_write_data(result_file,eeg_data,'dataformat','matlab');
% end
% alert()
