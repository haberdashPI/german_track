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
plot_cfg.preproc.detrend = 'no';
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subject 008

filename = eegfiles(1).name;
filepath = fullfile(raw_data_dir,filename);
numstr = regexp(filepath,'([0-9]+)_','tokens');
sid = str2num(numstr{1}{1});

% read in the events
event_file = fullfile(data_dir,sprintf('sound_events_%03d.csv',sid));
stim_events = readtable(event_file);

% read in eeg data header
[eeg,ntrials] = gt_loadbdf(filepath,stim_events,'lengths',sound_lengths,'channels',1:70);

eeg = gt_downsample(eeg,stim_events,8);
eeg = gt_settrials(@nt_demean,eeg);
eeg = gt_settrials(@gt_detrend,eeg,[1 10],'progress','detrending...');

% find bad channels, using linear detrending to avoid false positives
freq = 0.5;
bad_indices = gt_fortrials(@nt_find_bad_channels,eeg,freq,3,150,0.6,'channels',1:64);
eeg.hdr.label(bad_indices{1}) % run this line to see which indices are bad for a given trial

this_plot = plot_cfg;
this_plot.preproc.detrend = 'yes';
ft_databrowser(plot_cfg, eeg);

% interpolate bad channels
[closest,d]=nt_proximity('biosemi64.lay',63);
eeg = gt_settrials(@gt_interpolate_bad_channels,{eeg,bad_indices},closest,d,'channels',1:64);

% detrend again, this time recording the weights, for later use
[trials,w] = gt_fortrials(@gt_detrend,eeg,[1 10],'progress','detrending...');
eeg.trials = cellfun(@(x) x',trials,'UniformOutput',false);

% visualize the data
ft_databrowser(plot_cfg, eeg);

% find channel glitches (exclude ref and eye channels)
eegcat = gt_fortrials(@(x)x,eeg);
eegcat = vertcat(eegcat{:});
w = vertcat(w{:});
eegch = 1:64;
[outw,~] = gt_outliers(eegcat(:,eegch),w(:,eegch),2,3); % like nt_outliers, but shows a progress bar
nt_imagescc(outw')

this_plot = plot_cfg;
this_plot.continuous = 'yes';
ft_databrowser(this_plot, eeg);

eegcat(:,eegch)=gt_inpaint(eegcat(:,eegch),outw); % interpolate over outliers

% ft_databrowser(plot_cfg, eeg);

% step 1: find regions of likely eye blinks and movement
eog = 67:70; % the sensors near the eyes
time_shifts = -5:5;
eyes = eegcat(:,eog);
[B,A]=butter(2,1/(eeg.hdr.Fs/2), 'high');
tmp = filter(B,A,eyes);
plot(tmp(35000:36000,:));
% note: this participant has a strong alpha component in their activity
% I filter it out to find just eyeblinks
b = fir1(512,[8/(eeg.hdr.Fs/2) 14/(eeg.hdr.Fs/2)],'stop');
tmp2 = filtfilt(b,1,tmp);
figure; plot(tmp2(35000:36000,:));

pcas=nt_pca(tmp2,time_shifts,4);
figure; plot(pcas(35000:36000,:));
plot(((1:size(pcas))/eeg.hdr.Fs),pcas);
% blinks are quite rare in this subject
mask=abs(pcas(:,1))>8*median(abs(pcas(:,1)));
plot((1:size(eyes,1))/eeg.hdr.Fs,[tmp [mask; zeros(10,1)]*200])
% ... was this subject closing their eye to avoid eyeblinks (against my instructions???)

% step 2: find components using
C0=nt_cov(eegcat);
C1=nt_cov(bsxfun(@times,eegcat,[zeros(5,1);mask;zeros(5,1)]));
[todss,pwr0,pwr1] = nt_dss0(C0,C1);
% look at power of the components (to pick which ones to keep)
plot(pwr1./pwr0, '.-')
eye_comps = eegcat*todss(:,1:4);

% plot the components on a scalp
topo = [];
topo.component = 1:4;
topo.layout = lay;
ft_topoplotIC(topo,gt_ascomponent(eeg,todss));

% plot timecourse of the components
plot((1:size(eye_comps,1))/eeg.hdr.Fs,eye_comps)

% I think 1-2 will do the best jo2
eegclean = nt_tsr(eegcat,eye_comps(:,1:3),time_shifts);

% rereference
eegreref = nt_rereference(eegclean,[outw(6:end-5,:) ones(size(eegclean,1),6)]);

eegfinal = gt_asfieldtrip(eeg,eegreref,'cropfirst',5,'croplast',5);
ft_databrowser(plot_cfg, eegfinal);

savename = regexprep(filename,'.bdf$','.eeg');
save_subject_binary(eegfinal,fullfile(data_dir,savename))

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
cfg.dataset = filepath;
trial_lengths = round((sound_lengths(stim_events.sound_index)+0.5)*eeg.hdr.Fs);
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
eeg.hdr.Fs = eeg.hdr.Fs / 8;
eeg.time = gt_fortrials(@(data) ((1:size(data,1))/eeg.hdr.Fs),eeg)';
eeg.fsample = eeg.fsample / 8;
eeg.sampleinfo = round([stim_events.time*eeg.fsample stim_events.time*eeg.fsample + (cellfun(@(x) size(x,2),eeg.trial))' - 1]);

% detrend the trials
eeg = gt_settrials(@gt_detrend,eeg,[1 10],'progress','detrending...');

% find bad channels, using linear detrending to avoid false positives
freq = 0.5;
bad_indices = gt_fortrials(@nt_find_bad_channels,eeg,freq,3,150,0.75,'channels',1:64);
eeg.hdr.label(bad_indices{15}) % run this line to see which indices are bad for a given trial
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

% eyeblink removal

% step 1: find regions of likely eye blinks and movement
eog = 67:70; % the sensors near the eyes
time_shifts = -5:5;
eyes = eegcat(:,eog);
[B,A]=butter(2,1/(eeg.hdr.Fs/2), 'high');
plot(tmp);
tmp=nt_pca(tmp,time_shifts,4);
plot(((1:size(tmp,1))/eeg.hdr.Fs),tmp);
mask=abs(tmp(:,1))>3*median(abs(tmp(:,1)));
plot((1:size(eyes,1))/eeg.hdr.Fs,[eyes [mask; zeros(10,1)]*200])

% step 2: find components using
C0=nt_cov(eegcat);
C1=nt_cov(bsxfun(@times,eegcat,[zeros(5,1);mask;zeros(5,1)]));
[todss,pwr0,pwr1] = nt_dss0(C0,C1);
% look at power of the components (to pick which ones to keep)
plot(pwr1./pwr0, '.-')
eye_comps = eegcat*todss(:,1:8);

% plot the components on a scalp
topo = [];
topo.component = 1:8;
topo.layout = lay;
topo_data = eeg;
topo_data.topo = todss;
topo_data.unmixing = pinv(todss);
topo_data.topolabel = eeg.label; %cellfun(@(x)sprintf('EOG%02d',x),num2cell(1:size(todss,2)),'UniformOutput',0);
ft_topoplotIC(topo,topo_data);

% plot timecourse of the components
plot((1:size(eye_comps,1))/eeg.hdr.Fs,eye_comps)

% looks like 5 and 6 are good
eegclean = nt_tsr(eegcat,eye_comps(:,5:6),time_shifts);

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
save_subject_binary(eegfinal,fullfile(data_dir,savename))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subject 010

filename = eegfiles(3).name;
filepath = fullfile(raw_data_dir,filename);
numstr = regexp(filepath,'([0-9]+)_','tokens');
sid = str2num(numstr{1}{1});

% read in the events
event_file = fullfile(data_dir,sprintf('sound_events_%03d.csv',sid));
stim_events = readtable(event_file);

% read in eeg data header
[eeg,ntrials] = gt_loadbdf(filepath,stim_events,'lengths',sound_lengths,'channels',1:70);

eeg = gt_downsample(eeg,stim_events,8);
eeg = gt_settrials(@nt_demean,eeg);
eeg = gt_settrials(@gt_detrend,eeg,[1 10],'progress','detrending...');

% remove two channels I know, immediately are bad
eeg.hdr.label([28,57]) % shows the channels 'A28' and 'B25';
[closest,d]=nt_proximity('biosemi64.lay',63);
eeg = gt_settrials(@gt_interpolate_bad_channels,eeg,[28,57],closest,d,'channels',1:64);

% find bad channels, using linear detrending to avoid false positives
freq = 0.5;
bad_indices = gt_fortrials(@nt_find_bad_channels,eeg,freq,3,150,1.5,'channels',1:64);
eeg.hdr.label(bad_indices{1}) % run this line to see which indices are bad for a given trial

this_plot = plot_cfg;
this_plot.preproc.detrend = 'yes';
ft_databrowser(plot_cfg, eeg);

% interpolate bad channels
eeg = gt_settrials(@gt_interpolate_bad_channels,{eeg,bad_indices},closest,d,'channels',1:64);

% detrend again, this time recording the weights, for later use
[trials,w] = gt_fortrials(@gt_detrend,eeg,[1 10],'progress','detrending...');
eeg.trials = cellfun(@(x) x',trials,'UniformOutput',false);

% visualize the data
ft_databrowser(plot_cfg, eeg);

% find channel glitches (exclude ref and eye channels)
eegcat = gt_fortrials(@(x)x,eeg);
eegcat = vertcat(eegcat{:});
w = vertcat(w{:});
eegch = 1:64;
[outw,~] = gt_outliers(eegcat(:,eegch),w(:,eegch),2,3); % like nt_outliers, but shows a progress bar
nt_imagescc(outw')

this_plot = plot_cfg;
this_plot.continuous = 'yes';
ft_databrowser(this_plot, eeg);

eegcat(:,eegch)=gt_inpaint(eegcat(:,eegch),outw); % interpolate over outliers

% ft_databrowser(plot_cfg, eeg);

% step 1: find regions of likely eye blinks and movement
eog = 67:70; % the sensors near the eyes
time_shifts = -5:5;
eyes = eegcat(:,eog);
[B,A]=butter(2,1/(eeg.hdr.Fs/2), 'high');
tmp = filter(B,A,eyes);
figure; plot(tmp(8100:8800,:));
pcas=nt_pca(tmp,time_shifts,4);
figure; plot(((1:size(pcas(8100:8800,:)))/eeg.hdr.Fs),pcas(8100:8800,:));
% blinks are quite rare in this subject
mask=abs(pcas(:,2))>4*median(abs(pcas(:,2)));
plot((1:size(eyes,1))/eeg.hdr.Fs,[eyes [mask; zeros(10,1)]*200])
% ... was this subject closing their eye to avoid eyeblinks (against my instructions???)

% step 2: find components using
C0=nt_cov(eegcat);
C1=nt_cov(bsxfun(@times,eegcat,[zeros(5,1);mask;zeros(5,1)]));
[todss,pwr0,pwr1] = nt_dss0(C0,C1);
% look at power of the components (to pick which ones to keep)
plot(pwr1./pwr0, '.-')
eye_comps = eegcat*todss(:,1:7);

% plot the components on a scalp
topo = [];
topo.component = 1:7;
topo.layout = lay;
ft_topoplotIC(topo,gt_ascomponent(eeg,todss));

% plot timecourse of the components
plot((1:size(eye_comps,1))/eeg.hdr.Fs,eye_comps)

% I think 1-4 will do the best job
eegclean = nt_tsr(eegcat,eye_comps(:,1:4),time_shifts);

% rereference
eegreref = nt_rereference(eegclean,[outw(6:end-5,:) ones(size(eegclean,1),6)]);

eegfinal = gt_asfieldtrip(eeg,eegreref,'cropfirst',5,'croplast',5);
ft_databrowser(plot_cfg, eegfinal);

savename = regexprep(filename,'.bdf$','.eeg');
save_subject_binary(eegfinal,fullfile(data_dir,savename))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subject 011

filename = eegfiles(4).name;
filepath = fullfile(raw_data_dir,filename);
numstr = regexp(filepath,'([0-9]+)_','tokens');
sid = str2num(numstr{1}{1});

% read in the events
event_file = fullfile(data_dir,sprintf('sound_events_%03d.csv',sid));
stim_events = readtable(event_file);

% read in eeg data header
[eeg,ntrials] = gt_loadbdf(filepath,stim_events,'lengths',sound_lengths,'channels',1:70);

eeg = gt_downsample(eeg,stim_events,8);
eeg = gt_settrials(@nt_demean,eeg);
eeg = gt_settrials(@gt_detrend,eeg,[1 10],'progress','detrending...');

% find bad channels, using linear detrending to avoid false positives
freq = 0.5;
bad_indices = gt_fortrials(@nt_find_bad_channels,eeg,freq,3,150,1.5,'channels',1:64);
eeg.hdr.label(bad_indices{1}) % run this line to see which indices are bad for a given trial

ft_databrowser(plot_cfg, eeg);

% interpolate bad channels
eeg = gt_settrials(@gt_interpolate_bad_channels,{eeg,bad_indices},closest,d,'channels',1:64);

% detrend again, this time recording the weights, for later use
[trials,w] = gt_fortrials(@gt_detrend,eeg,[1 10],'progress','detrending...');
eeg.trials = cellfun(@(x) x',trials,'UniformOutput',false);

% visualize the data
ft_databrowser(plot_cfg, eeg);

% find channel glitches (exclude ref and eye channels)
eegcat = gt_fortrials(@(x)x,eeg);
eegcat = vertcat(eegcat{:});
w = vertcat(w{:});
eegch = 1:64;
[outw,~] = gt_outliers(eegcat(:,eegch),w(:,eegch),2,3); % like nt_outliers, but shows a progress bar
nt_imagescc(outw')

this_plot = plot_cfg;
this_plot.continuous = 'yes';
ft_databrowser(this_plot, eeg);

eegcat(:,eegch)=gt_inpaint(eegcat(:,eegch),outw); % interpolate over outliers

% ft_databrowser(plot_cfg, eeg);

% step 1: find regions of likely eye blinks and movement
eog = 67:70; % the sensors near the eyes
time_shifts = -5:5;
eyes = eegcat(:,eog);
[B,A]=butter(2,1/(eeg.hdr.Fs/2), 'high');
tmp = filter(B,A,eyes);
[B,A]=butter(2,30/(eeg.hdr.Fs/2), 'low');
tmp = filter(B,A,tmp);
figure; plot(tmp(1:2000,:));
pcas=nt_pca(tmp,time_shifts,4);
figure; plot(((1:size(pcas(1:2000,:)))/eeg.hdr.Fs),pcas(1:2000,:));
% blinks are quite rare in this subject
mask=abs(pcas(:,1))>2*median(abs(pcas(:,1)));
mask=min(1,mask + (abs(pcas(:,2))>5*median(abs(pcas(:,2)))));
plot((1:size(eyes,1))/eeg.hdr.Fs,[eyes [mask; zeros(10,1)]*200])
plot((1:size(eyes(1:2000,:),1))/eeg.hdr.Fs,[eyes(1:2000,:) mask(1:2000,:)*200 pcas(1:2000,:)])
% ... was this subject closing their eye to avoid eyeblinks (against my instructions???)

% step 2: find components using
C0=nt_cov(eegcat);
C1=nt_cov(bsxfun(@times,eegcat,[zeros(5,1);mask;zeros(5,1)]));
[todss,pwr0,pwr1] = nt_dss0(C0,C1);
% look at power of the components (to pick which ones to keep)
plot(pwr1./pwr0, '.-')
eye_comps = eegcat*todss(:,1:7);

% plot the components on a scalp
topo = [];
topo.component = 1:6;
topo.layout = lay;
ft_topoplotIC(topo,gt_ascomponent(eeg,todss));

% plot timecourse of the components
plot((1:size(eye_comps,1))/eeg.hdr.Fs,eye_comps)

% I think 1-5 will do the best job
eegclean = nt_tsr(eegcat,eye_comps(:,1:5),time_shifts);

% rereference
eegreref = nt_rereference(eegclean,[outw(6:end-5,:) ones(size(eegclean,1),6)]);

eegfinal = gt_asfieldtrip(eeg,eegreref,'cropfirst',5,'croplast',5);
ft_databrowser(plot_cfg, eegfinal);

savename = regexprep(filename,'.bdf$','.eeg');
save_subject_binary(eegfinal,fullfile(data_dir,savename))


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subject 012

filename = eegfiles(5).name;
filepath = fullfile(raw_data_dir,filename);
numstr = regexp(filepath,'([0-9]+)_','tokens');
sid = str2num(numstr{1}{1});

% read in the events
event_file = fullfile(data_dir,sprintf('sound_events_%03d.csv',sid));
stim_events = readtable(event_file);

% read in eeg data header
[eeg,ntrials] = gt_loadbdf(filepath,stim_events,'lengths',sound_lengths,'channels',1:70);

eeg = gt_downsample(eeg,stim_events,8);
eeg = gt_settrials(@nt_demean,eeg);
eeg = gt_settrials(@gt_detrend,eeg,[1 10],'progress','detrending...');

% remove two channels I know, immediately are bad
eeg.hdr.label([28]) % shows the channels 'A28' and 'B25';
[closest,d]=nt_proximity('biosemi64.lay',63);
eeg = gt_settrials(@gt_interpolate_bad_channels,eeg,[28,57],closest,d,'channels',1:64);

% find bad channels, using linear detrending to avoid false positives
freq = 0.5;
bad_indices = gt_fortrials(@nt_find_bad_channels,eeg,freq,3,150,1,'channels',1:64);
eeg.hdr.label(bad_indices{21}) % run this line to see which indices are bad for a given trial

ft_databrowser(plot_cfg, eeg);

% interpolate bad channels
eeg = gt_settrials(@gt_interpolate_bad_channels,{eeg,bad_indices},closest,d,'channels',1:64);

% detrend again, this time recording the weights, for later use
[trials,w] = gt_fortrials(@gt_detrend,eeg,[1 10],'progress','detrending...');
eeg.trials = cellfun(@(x) x',trials,'UniformOutput',false);

% visualize the data
ft_databrowser(plot_cfg, eeg);

% find channel glitches (exclude ref and eye channels)
eegcat = gt_fortrials(@(x)x,eeg);
eegcat = vertcat(eegcat{:});
w = vertcat(w{:});
eegch = 1:64;
[outw,~] = gt_outliers(eegcat(:,eegch),w(:,eegch),2,3); % like nt_outliers, but shows a progress bar
nt_imagescc(outw')

this_plot = plot_cfg;
this_plot.continuous = 'yes';
ft_databrowser(this_plot, eeg);

eegcat(:,eegch)=gt_inpaint(eegcat(:,eegch),outw); % interpolate over outliers

% ft_databrowser(plot_cfg, eeg);

% step 1: find regions of likely eye blinks and movement
eog = 67:70; % the sensors near the eyes
time_shifts = -5:5;
eyes = eegcat(:,eog);
[B,A]=butter(2,1/(eeg.hdr.Fs/2), 'high');
tmp = filter(B,A,eyes);
figure; plot(tmp(1:2000,:));
pcas=nt_pca(tmp,time_shifts,4);
plot(((1:size(pcas(1:2000,:)))/eeg.hdr.Fs),pcas(1:2000,:));
% blinks are quite rare in this subject
mask=abs(pcas(:,1))>4*median(abs(pcas(:,1)));
plot((1:size(eyes,1))/eeg.hdr.Fs,[eyes [mask; zeros(10,1)]*200])
plot((1:size(eyes(1:2000,:),1))/eeg.hdr.Fs,[eyes(1:2000,:) mask(1:2000,:)*200 pcas(1:2000,:)])
% ... was this subject closing their eye to avoid eyeblinks (against my instructions???)

% step 2: find components using
C0=nt_cov(eegcat);
C1=nt_cov(bsxfun(@times,eegcat,[zeros(5,1);mask;zeros(5,1)]));
[todss,pwr0,pwr1] = nt_dss0(C0,C1);
% look at power of the components (to pick which ones to keep)
plot(pwr1./pwr0, '.-')
eye_comps = eegcat*todss(:,1:7);

% plot the components on a scalp
topo = [];
topo.component = 1:5;
topo.layout = lay;
ft_topoplotIC(topo,gt_ascomponent(eeg,todss));

% plot timecourse of the components
plot((1:size(eye_comps,1))/eeg.hdr.Fs,eye_comps)

% I think 1-5 will do the best job
eegclean = nt_tsr(eegcat,eye_comps(:,4),time_shifts);

% rereference
eegreref = nt_rereference(eegclean,[outw(6:end-5,:) ones(size(eegclean,1),6)]);

eegfinal = gt_asfieldtrip(eeg,eegreref,'cropfirst',5,'croplast',5);
ft_databrowser(plot_cfg, eegfinal);

savename = regexprep(filename,'.bdf$','.eeg');
save_subject_binary(eegfinal,fullfile(data_dir,savename))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subject 013

filename = eegfiles(6).name;
filepath = fullfile(raw_data_dir,filename);
numstr = regexp(filepath,'([0-9]+)_','tokens');
sid = str2num(numstr{1}{1});

% read in the events
event_file = fullfile(data_dir,sprintf('sound_events_%03d.csv',sid));
stim_events = readtable(event_file);

% read in eeg data header
[eeg,ntrials] = gt_loadbdf(filepath,stim_events,'lengths',sound_lengths,'channels',1:70);
% useful when we concatenate the trials for analysis
trial_markers = cumsum(eeg.sampleinfo(:,2) - eeg.sampleinfo(:,1));

eeg = gt_downsample(eeg,stim_events,8);
eeg = gt_settrials(@nt_demean,eeg);
eeg = gt_settrials(@gt_detrend,eeg,[1 10],'progress','detrending...');

% remove two channels I know, immediately are bad
eeg.hdr.label([28]); % shows the channels 'A28' and 'B25';
[closest,d]=nt_proximity('biosemi64.lay',63);
eeg = gt_settrials(@gt_interpolate_bad_channels,eeg,28,closest,d,'channels',1:64);

% find bad channels, using linear detrending to avoid false positives
freq = 0.5;
bad_indices = gt_fortrials(@nt_find_bad_channels,eeg,freq,3,150,2,'channels',1:64);
eeg.hdr.label(bad_indices{21}) % run this line to see which indices are bad for a given trial

ft_databrowser(plot_cfg, eeg);

% interpolate bad channels
eeg = gt_settrials(@gt_interpolate_bad_channels,{eeg,bad_indices},closest,d,'channels',1:64);

% detrend again, this time recording the weights, for later use
[trials,w] = gt_fortrials(@gt_detrend,eeg,[1 10],'progress','detrending...');
eeg.trials = cellfun(@(x) x',trials,'UniformOutput',false);

% visualize the data
ft_databrowser(plot_cfg, eeg);

% find channel glitches (exclude ref and eye channels)
eegcat = gt_fortrials(@(x)x,eeg);
eegcat = vertcat(eegcat{:});
w = vertcat(w{:});
eegch = 1:64;
[outw,~] = gt_outliers(eegcat(:,eegch),w(:,eegch),2,3); % like nt_outliers, but shows a progress bar
nt_imagescc(outw')

this_plot = plot_cfg;
this_plot.continuous = 'yes';
ft_databrowser(this_plot, eeg);

eegcat(:,eegch)=gt_inpaint(eegcat(:,eegch),outw); % interpolate over outliers

% ft_databrowser(plot_cfg, eeg);

% step 1: find regions of likely eye blinks and movement
eog = 67:70; % the sensors near the eyes
time_shifts = -5:5;
eyes = eegcat(:,eog);
[B,A]=butter(2,1/(eeg.hdr.Fs/2), 'high');
tmp = filter(B,A,eyes);
% select trial 8 as a good example
trial = trial_markers(13):trial_markers(14);
% figure;
plot(tmp(trial,:));
pcas=nt_pca(tmp,time_shifts,4);
plot(((1:size(pcas(trial,:)))/eeg.hdr.Fs),pcas(trial,:));
% blinks are quite rare in this subject
mask=abs(pcas(:,1))>4*median(abs(pcas(:,1)));
plot((1:size(eyes,1))/eeg.hdr.Fs,[eyes [mask; zeros(10,1)]*200])
plot((1:size(eyes(trial,:),1))/eeg.hdr.Fs,[eyes(trial,:) mask(trial,:)*200 pcas(trial,:)])
% ... was this subject closing their eye to avoid eyeblinks (against my instructions???)

% step 2: find components using
C0=nt_cov(eegcat);
C1=nt_cov(bsxfun(@times,eegcat,[zeros(5,1);mask;zeros(5,1)]));
[todss,pwr0,pwr1] = nt_dss0(C0,C1);
% look at power of the components (to pick which ones to keep)
figure; plot(pwr1./pwr0, '.-')
eye_comps = eegcat*todss(:,1:5);

% plot the components on a scalp
topo = [];
topo.component = 1:5;
topo.layout = lay;
ft_topoplotIC(topo,gt_ascomponent(eeg,todss));

% plot timecourse of the components
figure; plot((1:size(eye_comps(trial,:),1))/eeg.hdr.Fs,eye_comps(trial,:))

% I think 1-5 will do the best job
eegclean = nt_tsr(eegcat,eye_comps(:,1:5),time_shifts);

% rereference
eegreref = nt_rereference(eegclean,[outw(6:end-5,:) ones(size(eegclean,1),6)]);

eegfinal = gt_asfieldtrip(eeg,eegreref,'cropfirst',5,'croplast',5);
ft_databrowser(plot_cfg, eegfinal);

savename = regexprep(filename,'.bdf$','.eeg');
save_subject_binary(eegfinal,fullfile(data_dir,savename))

