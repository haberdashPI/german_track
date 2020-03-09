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
plot_cfg.blocksize = 10;

% topographic alyout
elec = ft_read_sens(fullfile(raw_data_dir,'biosemi64.txt'));
cfg = [];
cfg.elec = elec;
lay = ft_prepare_layout(cfg);
ft_layoutplot(cfg)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% find the length of all stimuli

soundsdir = fullfile(stim_data_dir,'mixtures','testing');
sounds = sort({dir(fullfile(soundsdir,'*.wav')).name});
sound_lengths = zeros(length(sounds),1);
for i = 1:length(sounds)
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

filename = eegfiles(2).name;
filepath = fullfile(raw_data_dir,filename);
numstr = regexp(filepath,'([0-9]+)_','tokens');
sid = str2num(numstr{1}{1});

% read in the events
event_file = fullfile(data_dir,sprintf('sound_events_%03d.csv',sid));
stim_events = readtable(event_file);

% read in eeg data header
head = ft_read_header(filepath);
% for SID 9, we recorded some extra channels (skip them)
head.label = head.label([1:64 129:134]);
head.nChans = 70;

% define trial boundaries
cfg = [];
cfg.dataset = filepath
trial_lengths = round((sound_lengths(stim_events.sound_index)+0.5)*head.Fs);
cfg.trl = [stim_events.sample stim_events.sample + trial_lengths ...
           zeros(length(stim_events.sample),1)];

% read in the data
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

% detrend the trials
eeg = gt_settrials(@gt_detrend,eeg,[1 10],'progress','detrending...');

% find bad channels, using linear detrending to avoid false positives
freq = 0.5;
bad_indices = gt_fortrials(@nt_find_bad_channels,eeg,freq,3,150,0.75,'channels',1:64);
head.label(bad_indices{15}) % run this line to see which indices are bad for a given trial
ft_databrowser(plot_cfg, eeg);

% interpolate bad channels
[closest,d]=nt_proximity('biosemi64.lay',63);
eeg = gt_settrials(@gt_interpolate_bad_channels,{eeg,bad_indices},closest,d,'channels',1:64);

% visualize the data
ft_databrowser(plot_cfg, eeg);

% detrend again, this time recording the weights, for later use
[trials,w] = gt_fortrials(@gt_detrend,eeg,[1 10],'progress','detrending...');
eeg.trials = cellfun(@(x) x',trials,'UniformOutput',false);
this_plot = plot_cfg;
this_plot.preproc.detrend = 'no';
ft_databrowser(this_plot, eeg);

% find channel glitches (exclude ref and eyeblinks)
eegcat = gt_fortrials(@(x)x,eeg);
eegcat = vertcat(eegcat{:});
w = vertcat(w{:});
eegch = 1:64;
[outw,y] = gt_outliers(eegcat(:,eegch),w(:,eegch),2,3); % like nt_outliers, but shows a progress bar
nt_imagescc(outw')

eegcat(:,eegch)=gt_inpaint(eegcat(:,eegch),outw); % interpolate over outliers

% find glithes in eye and reference channels
exch= 65:70
[outexw,exy] = gt_outliers(eegcat(:,exch),w(:,exch),2,3); % like nt_outliers, but shows a progress bar
eegcat(:,exch)=gt_inpaint(eegcat,outexw); % interpolate over outliers
nt_imagescc(outexw')

eegcat(:,exch)=gt_inpaint(eegcat(:,exch),outexw); % interpolate over outliers

% eyeblink removal

% step 1: find regions of likely eye blinks and movement
eog = 67:70; % the sensors near the eyes
time_shifts = -5:5
eyes = eegcat(:,eog);
[B,A]=butter(2,1/(head.Fs/2), 'high');
tmp=nt_pca(filter(B,A,eyes),time_shifts,4);
plot(((1:size(tmp,1))/head.Fs),tmp);
[b,a]=butter(5,20/(head.Fs/2), 'low');
tmp=filter(b,a,tmp);
plot(((1:size(tmp,1))/head.Fs),tmp);
mask=abs(tmp(:,1))>3*median(abs(tmp(:,1)));
plot((1:size(eyes,1))/head.Fs,[eyes [mask; zeros(10,1)]*200])

% step 2: find components using
C0=nt_cov(eegcat);
C1=nt_cov(bsxfun(@times,eegcat,[zeros(5,1);mask;zeros(5,1)]));
[todss,pwr0,pwr1] = nt_dss0(C0,C1);
% look at power of the components (to pick which ones to keep)
plot(pwr1./pwr0, '.-')
eye_comps = eegcat*todss(:,1:5);
plot((1:size(eye_comps,1))/head.Fs,eye_comps)
% Component 4 doesn't look very convincing
plot((1:size(eye_comps,1))/head.Fs,eye_comps(:,1:3))

% plot the components on a scalp to verify
topo = [];
topo.component = 1:4;
topo.layout = lay;
topo_data = eeg;
topo_data.topo = todss;
topo_data.unmixing = pinv(todss);
topo_data.topolabel = eeg.label; %cellfun(@(x)sprintf('EOG%02d',x),num2cell(1:size(todss,2)),'UniformOutput',0);
ft_topoplotIC(topo,topo_data);

eegclean = nt_tsr(eegcat,eye_comps(:,1:4),time_shifts);

% rereference
eegreref = nt_rereference(eegclean,[outw(6:end-5,:) outexw(6:end-5,:)]);

eegfinal = eeg;
eegfinal.time{1} = eegfinal.time{1}(6:end);
eegfinal.time{end} = eegfinal.time{end}(1:end-5);
eegfinal.trial = {};
eegfinal.sampleinfo = eeg.sampleinfo;
eegfinal.sampleinfo(1,1) = eegfinal.sampleinfo(1,1)+5;
eegfinal.sampleinfo(end,2) = eegfinal.sampleinfo(end,2)-5;
k = 1;
for i = 1:ntrials
    n = size(eeg.trial{i},2);
    if i == 1 || i == ntrials
        n = n - 5;
    end
    eegfinal.trial{i} = eegreref(k:(k+n-1),:)';
    k = k+n;
end

this_plot = plot_cfg;
this_plot.preproc.detrend = 'no';
ft_databrowser(plot_cfg, eegfinal);

savename = regexprep(filename,'.bdf$','.eeg');
save_subject_binary(eegfinal,fullfile(data_dir,name))
