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

[closest,dists]=nt_proximity('biosemi64.lay',63);

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
% setup parameters for individual participant

% Each subject should have a different entry belowto tune the data cleaning
% parameters to their data.

subject = [];

subject(1).sid = 8;
subject(1).load_channels = 1:70;
subject(1).known_bad_channels = [];
subject(1).bad_channel_threshs = {3,150,0.6};
subject(1).fix_glitches = false;
subject(1).trial_outlier_threshs = [];
subject(1).crop_eye_components = 0.25;
subject(1).eye_pca_comps = 1;
subject(1).eye_mask_threshold = 6;
subject(1).plot_eye_comps = 1:3;
subject(1).eye_comps = 1:2;

subject(2).sid = 9;
subject(2).load_channels = [1:64,129:134];
subject(2).known_bad_channels = 28;
subject(2).bad_channel_threshs = {3,150,1};
subject(2).fix_glitches = false;
subject(2).trial_outlier_threshs = [];
subject(2).crop_eye_components = 0.5;
subject(2).eye_pca_comps = 1;
subject(2).eye_mask_threshold = 4;
subject(2).plot_eye_comps = 1:4;
subject(2).eye_comps = 1:2;

subject(3).sid = 10;
subject(3).load_channels = 1:70;
subject(3).known_bad_channels = [28,57];
subject(3).bad_channel_threshs = {3,150,1};
subject(3).fix_glitches = false;
subject(3).trial_outlier_threshs = [];
subject(3).crop_eye_components = 0.5;
subject(3).eye_pca_comps = 2;
subject(3).eye_mask_threshold = 6;
subject(3).plot_eye_comps = 1:3;
subject(3).eye_comps = 1:2;

subject(4).sid = 11;
subject(4).load_channels = 1:70;
subject(4).known_bad_channels = [28];
subject(4).bad_channel_threshs = {2,150,1};
subject(4).fix_glitches = true;
subject(4).glitch_params = {6,4};
subject(4).trial_outlier_threshs = [4,3];
subject(4).crop_eye_components = 0.5;
subject(4).eye_pca_comps = 2;
subject(4).eye_mask_threshold = 4;
subject(4).plot_eye_comps = 1:4;
subject(4).eye_comps = 1:4;

subject(5).sid = 12;
subject(5).load_channels = 1:70;
subject(5).known_bad_channels = [28];
subject(5).bad_channel_threshs = {3,150,1};
subject(5).fix_glitches = false;
subject(5).trial_outlier_threshs = [];
subject(5).crop_eye_components = 0.5;
subject(5).eye_pca_comps = 2;
subject(5).eye_mask_threshold = 5;
subject(5).plot_eye_comps = 1:3;
subject(5).eye_comps = 1:1;

subject(6).sid = 13;
subject(6).load_channels = 1:70;
subject(6).known_bad_channels = [28];
subject(6).bad_channel_threshs = {2,150,1};
subject(6).fix_glitches = true;
subject(6).glitch_params = {8,4};
subject(6).trial_outlier_threshs = 4;
subject(6).crop_eye_components = [];
subject(6).eye_pca_comps = 1;
subject(6).eye_mask_threshold = 6;
subject(6).plot_eye_comps = 1:4;
subject(6).eye_comps = 1:4;

subject(7).sid = 14;
subject(7).load_channels = 1:70;
subject(7).known_bad_channels = [28];
subject(7).bad_channel_threshs = {2,150,1};
subject(7).fix_glitches = false;
subject(7).trial_outlier_threshs = [];
subject(7).crop_eye_components = 0.5;
subject(7).eye_pca_comps = 1;
subject(7).eye_mask_threshold = 4;
subject(7).plot_eye_comps = 1:4;
subject(7).eye_comps = 1:2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% if you are rerunning analyses you can set this to false, if you are analyzing a
% new subject set this to true and run each section below, one at a time, using
% the plots to verify the results
interactive = false;
for i = 1:length(eegfiles)

    %% file information
    filename = eegfiles(i).name;
    filepath = fullfile(raw_data_dir,filename);
    numstr = regexp(filepath,'([0-9]+)_','tokens');
    sid = str2num(numstr{1}{1});
    if subject(i).sid ~= sid
        error('Wrong subject id (%d) specified for index %d, expected sid %d',...
            subject(i).sid,i,sid);
    end

    %% read in the events
    event_file = fullfile(data_dir,sprintf('sound_events_%03d.csv',sid));
    stim_events = readtable(event_file);

    %% read in eeg data header
    eeg = gt_loadbdf(filepath,stim_events,'lengths',sound_lengths,'channels',subject(i).load_channels);

    %% preprocess data
    eeg = gt_downsample(eeg,stim_events,8);
    eeg = gt_settrials(@nt_demean,eeg);
    if interactive
        ft_databrowser(plot_cfg, eeg);
    end

    eeg = gt_settrials(@gt_interpolate_bad_channels,eeg,...
        subject(i).known_bad_channels,closest,dists,'channels',1:64);
    [eeg,w] = gt_settrials(@gt_detrend,eeg,[1 10],'progress','detrending...');

    %% find bad channels
    freq = 0.5;
    bad_indices = gt_fortrials(@nt_find_bad_channels,eeg,freq,...
        subject(i).bad_channel_threshs{:},'channels',1:64);
    if interactive
        eeg.hdr.label(bad_indices{8}) % run this line to see which indices are bad for a given trial
        this_plot = plot_cfg;
        this_plot.preproc.detrend = 'yes';
        ft_databrowser(plot_cfg, eeg);
    end

    %% interpolate bad channels
    eeg = gt_settrials(@gt_interpolate_bad_channels,{eeg,bad_indices},closest,dists,'channels',1:64);
    if interactive
        ft_databrowser(plot_cfg, eeg);
    end

    % find single-channel glitches
    if subject(i).fix_glitches

        % question, do I need to do this on a per trial basis?
        [eegoutl,w] = gt_settrials(@gt_outliers,{eeg,w},...
            subject(i).glitch_params{:},false,'progress','finding outliers...'); % like nt_outliers, but shows a progress bar
        if interactive
            ft_databrowser(plot_cfg, eegoutl);
            ft_databrowser(plot_cfg, gt_asfieldtrip(eegoutl,w));
        end

        eeg = eegoutl;
    end

    %% find outlier trials
    for thresh = subject(i).trial_outlier_threshs

        keep = nt_find_outlier_trials(nt_trial2mat(eeg.trial,round(max(sound_lengths)*eeg.hdr.Fs)),thresh);
        discard = setdiff(1:length(eeg.trial),keep);

        for d = discard
            eeg.trial{d} = zeros(size(eeg.trial{d}));
            w{d} = zeros(size(w{d}));
        end

    end

    % concatenate all trials
    eegcat = gt_fortrials(@(x)x,eeg);
    eegcat = vertcat(eegcat{:});
    wcat = vertcat(w{:});

    eegreref = nt_rereference(eegcat,wcat);
    if interactive
        ft_databrowser(plot_cfg, gt_asfieldtrip(eeg,eegreref));
    end

    %% select and filter eyeblink channels
    eog = 67:70; % the sensors near the eyes
    time_shifts = 0:10;
    eyes = eegreref(:,eog);
    [B,A]=butter(2,1/(eeg.hdr.Fs/2), 'high');
    tmp = filter(B,A,eyes);
    b = fir1(512,[8/(eeg.hdr.Fs/2) 14/(eeg.hdr.Fs/2)],'stop');
    tmp= filtfilt(b,1,tmp);
    if interactive
        chans = cellfun(@(x)sprintf('eog%02d',x),num2cell(1:length(eog)),...
            'UniformOutput',false);
        this_plot = plot_cfg;
        this_plot.ylim = [-100 100];
        ft_databrowser(this_plot, gt_asfieldtrip(eeg,tmp,'label',chans))
    end

    %% compute TSPCAs (crop the start and end of each trial, because it is glitchy)
    if ~isempty(subject(i).crop_eye_components)
        len = subject(i).crop_eye_components;
        pcaweight = gt_fortrials(@(x)gt_cropend_weights(x,round(eeg.hdr.Fs*len)),eeg);
        pcaweight = vertcat(pcaweight{:});
    else
        pcaweight = 1;
    end
    [pcas,idx]=nt_pca(tmp.*pcaweight,time_shifts,4);
    if interactive
        chans = cellfun(@(x)sprintf('pc%02d',x),num2cell(1:4),'UniformOutput',false);
        this_plot = plot_cfg;
        this_plot.ylim = [-400 400];
        ft_databrowser(this_plot, gt_asfieldtrip(eeg,pcas,'label',chans,'cropfirst',10));
    end

    %% compute a mask, to select regions of probable eyeblinks
    c = abs(hilbert(pcas(:,subject(i).eye_pca_comps)));
    mask=abs(c)>subject(i).eye_mask_threshold*median(abs(c));
    if interactive
        eyemask = [eyes [mask; zeros(10,1)]*200];
        chans = cellfun(@(x)sprintf('eye%02d',x),num2cell(1:4),'UniformOutput',false);
        chans = [ chans 'mask' ];
        this_plot = plot_cfg;
        this_plot.ylim = [-200 200];
        ft_databrowser(this_plot, gt_asfieldtrip(eeg,eyemask,'label',chans))
    end

    %% compute eyeblink sources
    C0=nt_cov(eegreref);
    C1=nt_cov(bsxfun(@times,eegreref,[zeros(5,1);mask;zeros(5,1)]));
    [todss,pwr0,pwr1] = nt_dss0(C0,C1);
    % look at power of the components (to pick which ones to keep)

    %% plot components in several ways to verify
    if interactive
        plot(pwr1./pwr0, '.-')

        comps = subject(i).plot_eye_comps;
        eye_comps = eegreref*todss(:,comps);
        topo = [];
        topo.component = comps;
        topo.layout = lay;
        ft_topoplotIC(topo,gt_ascomponent(eeg,todss(:,comps)));

        % plot timecourse of components
        chans = cellfun(@(x)sprintf('eye%02d',x),num2cell(comps),'UniformOutput',false);
        this_plot = plot_cfg;
        this_plot.ylim = [-1 1] .* 1e-1;
        ft_databrowser(this_plot, gt_asfieldtrip(eeg,eye_comps,'label',chans,'croplast',10))

    end

    %% apply eye blinks to clean data, and rereference
    eye_comps = eegreref*todss(:,subject(i).eye_comps);
    eegfinal = gt_asfieldtrip(eeg,nt_tsr(eegreref,eye_comps,time_shifts),'croplast',10);
    if interactive
        ft_databrowser(plot_cfg, eegfinal);
    end

    % save the results
    savename = regexprep(filename,'.bdf$','.eeg');
    save_subject_binary(eegfinal,fullfile(data_dir,savename));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subject 009
%{
% get filename and SID

filename = eegfiles(2).name;
filepath = fullfile(raw_data_dir,filename);
numstr = regexp(filepath,'([0-9]+)_','tokens');
sid = str2num(numstr{1}{1});

% read in the events
event_file = fullfile(data_dir,sprintf('sound_events_%03d.csv',sid));
stim_events = readtable(event_file);

eeg = gt_loadbdf(filepath,stim_events,'lengths',sound_lengths,'channels',[1:64,129:134]);

% downsample, demean and detrend the data
eeg = gt_downsample(eeg,stim_events,8);
eeg = gt_settrials(@nt_demean,eeg);
eeg = gt_settrials(@gt_detrend,eeg,[1 10],'progress','detrending...');

% remove two channels I know, immediately are bad
eeg.hdr.label([28]) % shows the channels 'A28' and 'B25';
[closest,d]=nt_proximity('biosemi64.lay',63);
eeg = gt_settrials(@gt_interpolate_bad_channels,eeg,[28],closest,d,'channels',1:64);

% find bad channels, using linear detrending to avoid false positives
freq = 0.5;
bad_indices = gt_fortrials(@nt_find_bad_channels,eeg,freq,3,150,1,'channels',1:64);
eeg.hdr.label(bad_indices{1})
ft_databrowser(plot_cfg, eeg);

% interpolate bad channels
[closest,d]=nt_proximity('biosemi64.lay',63);
eeg = gt_settrials(@gt_interpolate_bad_channels,{eeg,bad_indices},closest,d,'channels',1:64);

% visualize the data
ft_databrowser(plot_cfg, eeg);

% detrend again, this time recording the weights, for later use
[trials,w] = gt_fortrials(@gt_detrend,eeg,[1 10],'progress','detrending...');
eeg.trials = cellfun(@(x) x',trials,'UniformOutput',false);
ft_databrowser(plot_cfg, eeg);

% % find channel glitches (exclude ref and eyeblinks)
% eegcat = gt_fortrials(@(x)x,eeg);
% eegcat = vertcat(eegcat{:});
% w = vertcat(w{:});
% eegch = 1:64;
% [outw,y] = gt_outliers(eegcat(:,eegch),w(:,eegch),2,3); % like nt_outliers, but shows a progress bar

% % inspect the weights
% this_plot = plot_cfg;
% this_plot.ylim = [0 1];
% ft_databrowser(this_plot,gt_asfieldtrip(eeg,[outw zeros(size(outw,1),6)]));
% ft_databrowser(plot_cfg, eeg);

% eegcat(:,eegch)=gt_inpaint(eegcat(:,eegch),outw); % interpolate over outliers

% eyeblink removal

% step 1: find regions of likely eye blinks and movement
eog = 67:70; % the sensors near the eyes
time_shifts = -5:5;
eyes = eegcat(:,eog);
[B,A]=butter(2,1/(eeg.hdr.Fs/2), 'high');

w = gt_fortrials(@(x) gt_gropend_weights(x,round(eeg.hdr.Fs*)))
tmp=nt_pca(tmp,time_shifts,4,[],[],);

% figure;
chans = cellfun(@(x)sprintf('pca%02d',x),num2cell(1:4),'UniformOutput',false);
this_plot = plot_cfg;
this_plot.ylim = [-30 30];
ft_databrowser(this_plot, gt_asfieldtrip(eeg,pcas,'label',chans,...
    'cropfirst',5,'croplast',5))

% compute signal envelope via hilbert transform
c = pcas(:,1);
c = abs(hilbert(c));
mask=abs(c)>5*median(abs(c));

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
trial = markers(13):markers(14);
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subject 014

filename = eegfiles(7).name;
filepath = fullfile(raw_data_dir,filename);
numstr = regexp(filepath,'([0-9]+)_','tokens');
sid = str2num(numstr{1}{1});

% read in the events
event_file = fullfile(data_dir,sprintf('sound_events_%03d.csv',sid));
stim_events = readtable(event_file);

% read in eeg data header
[eeg,ntrials] = gt_loadbdf(filepath,stim_events,'lengths',sound_lengths,'channels',1:70);
% useful when we concatenate the trials for analysis

eeg = gt_downsample(eeg,stim_events,8);
eeg = gt_settrials(@nt_demean,eeg);
eeg = gt_settrials(@gt_detrend,eeg,[1 10],'progress','detrending...');
markers = cumsum(eeg.sampleinfo(:,2) - eeg.sampleinfo(:,1));

% remove two channels I know, immediately are bad
eeg.hdr.label([28]) % shows the channels 'A28' and 'B25';
[closest,d]=nt_proximity('biosemi64.lay',63);
eeg = gt_settrials(@gt_interpolate_bad_channels,eeg,[28],closest,d,'channels',1:64);

% find bad channels, using linear detrending to avoid false positives
freq = 0.5;
bad_indices = gt_fortrials(@nt_find_bad_channels,eeg,freq,2,150,1,'channels',1:64);
eeg.hdr.label(bad_indices{1}) % run this line to see which indices are bad for a given trial
eeg.hdr.label(bad_indices{2}) % run this line to see which indices are bad for a given trial

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
% there are basically no glithes
% w = vertcat(w{:});
% eegch = 1:64;
% [outw,~] = gt_outliers(eegcat(:,eegch),w(:,eegch),2,3); % like nt_outliers, but shows a progress bar

% % inspect the weights
% this_plot = plot_cfg;
% this_plot.ylim = [0 1];
% ft_databrowser(this_plot,gt_asfieldtrip(eeg,[outw zeros(size(outw,1),6)]));
% figure; ft_databrowser(plot_cfg, eeg);

% eegcat(:,eegch)=gt_inpaint(eegcat(:,eegch),outw); % interpolate over outliers

% ft_databrowser(plot_cfg, eeg);

% visualize the data
figure; ft_databrowser(plot_cfg, gt_asfieldtrip(eeg,eegcat));
f2 = figure;

% step 1: find regions of likely eye blinks and movement
eog = 67:70; % the sensors near the eyes
time_shifts = -5:5;
eyes = eegcat(:,eog);
[B,A]=butter(2,1/(eeg.hdr.Fs/2), 'high');
tmp = filter(B,A,eyes);
trial = markers(4):markers(5);
% figure;
figure(f2); plot(tmp(trial,:));
pcas=nt_pca(tmp,time_shifts,4);
figure(f2); plot(((1:size(pcas(trial,:)))/eeg.hdr.Fs),pcas(trial,:));
% blinks are quite rare in this subject
mask=abs(pcas(:,1))>5*median(abs(pcas(:,1)));
plot((1:size(eyes,1))/eeg.hdr.Fs,[eyes [mask; zeros(10,1)]*200])
plot((1:size(eyes(trial,:),1))/eeg.hdr.Fs,[eyes(trial,:) mask(trial,:)*200 pcas(trial,:)])
% ... was this subject closing their eye to avoid eyeblinks (against my instructions???)

% step 2: find components using
C0=nt_cov(eegcat);
C1=nt_cov(bsxfun(@times,eegcat,[zeros(5,1);mask;zeros(5,1)]));
[todss,pwr0,pwr1] = nt_dss0(C0,C1);
% look at power of the components (to pick which ones to keep)
figure; plot(pwr1./pwr0, '.-')
eye_comps = eegcat*todss(:,1:6);

% plot the components on a scalp
topo = [];
topo.component = 1:6;
topo.layout = lay;
ft_topoplotIC(topo,gt_ascomponent(eeg,todss));

% plot timecourse of the components
chans = cellfun(@(x)sprintf('eye%02d',x),num2cell(1:6),'UniformOutput',false);
this_plot = plot_cfg;
this_plot.ylim = [-1e-3 1e-3];
figure; ft_databrowser(this_plot, gt_asfieldtrip(eeg,eye_comps,'label',chans))

eegclean = nt_tsr(eegcat,eye_comps(:,1:3),time_shifts);

% rereference
eegreref = nt_rereference(eegclean,[outw(6:end-5,:) ones(size(eegclean,1),6)]);

eegfinal = gt_asfieldtrip(eeg,eegreref,'cropfirst',5,'croplast',5);
ft_databrowser(plot_cfg, eegfinal);

savename = regexprep(filename,'.bdf$','.eeg');
save_subject_binary(eegfinal,fullfile(data_dir,savename))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subject 016

filename = eegfiles(8).name;
filepath = fullfile(raw_data_dir,filename);
numstr = regexp(filepath,'([0-9]+)_','tokens');
sid = str2num(numstr{1}{1});

% read in the events
event_file = fullfile(data_dir,sprintf('sound_events_%03d.csv',sid));
stim_events = readtable(event_file);

% read in eeg data header
[eeg,ntrials] = gt_loadbdf(filepath,stim_events,'lengths',sound_lengths,'channels',1:70);
% useful when we concatenate the trials for analysis

eeg = gt_downsample(eeg,stim_events,8);
eeg = gt_settrials(@nt_demean,eeg);
eeg = gt_settrials(@gt_detrend,eeg,[1 10],'progress','detrending...');
markers = cumsum(eeg.sampleinfo(:,2) - eeg.sampleinfo(:,1));

% remove two channels I know, immediately are bad
eeg.hdr.label([4]) % shows the channels 'A28' and 'B25';
[closest,d]=nt_proximity('biosemi64.lay',63);
eeg = gt_settrials(@gt_interpolate_bad_channels,eeg,[4],closest,d,'channels',1:64);

freq = 0.5;
bad_indices = gt_fortrials(@nt_find_bad_channels,eeg,freq,3,150,0.8,'channels',1:64);
eeg.hdr.label(bad_indices{104}) % run this line to see which indices are bad for a given trial

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
% there are basically no glithes (after agressive channel imterpolation)
% w = vertcat(w{:});
% eegch = 1:64;
% [outw,~] = gt_outliers(eegcat(:,eegch),w(:,eegch),2,3); % like nt_outliers, but shows a progress bar

% % inspect the weights
% this_plot = plot_cfg;
% this_plot.ylim = [0 1];
% ft_databrowser(this_plot,gt_asfieldtrip(eeg,[outw zeros(size(outw,1),6)]));
% figure; ft_databrowser(plot_cfg, eeg);

% eegcat(:,eegch)=gt_inpaint(eegcat(:,eegch),outw); % interpolate over outliers

% ft_databrowser(plot_cfg, eeg);

% visualize the data
figure; ft_databrowser(plot_cfg, gt_asfieldtrip(eeg,eegcat));
f2 = figure;


% step 1: find regions of likely eye blinks and movement
eog = 67:70; % the sensors near the eyes
time_shifts = -5:5;
eyes = eegcat(:,eog);
[B,A]=butter(2,1/(eeg.hdr.Fs/2), 'high');
tmp = filter(B,A,eyes);
pcas=nt_pca(tmp,time_shifts,4);

% figure;
chans = cellfun(@(x)sprintf('pca%02d',x),num2cell(1:4),'UniformOutput',false);
this_plot = plot_cfg;
this_plot.ylim = [-30 30];
figure; ft_databrowser(this_plot, gt_asfieldtrip(eeg,pcas,'label',chans,...
    'cropfirst',5,'croplast',5))

% compute signal envelope via hilbert transform
c = pcas(:,2);
c = abs(hilbert(c));

% blinks are quite rare in this subject
mask=abs(c)>10*median(abs(pcas(:,2)));
eyemask = [eyes [mask; zeros(10,1)]*200];
chans = cellfun(@(x)sprintf('eye%02d',x),num2cell(1:4),'UniformOutput',false);
chans = [ chans 'mask' ];
this_plot = plot_cfg;
this_plot.ylim = [-200 200];
ft_databrowser(this_plot, gt_asfieldtrip(eeg,eyemask,'label',chans))


% ... was this subject closing their eye to avoid eyeblinks (against my instructions???)

% step 2: find components using
C0=nt_cov(eegcat);
C1=nt_cov(bsxfun(@times,eegcat,[zeros(5,1);mask;zeros(5,1)]));
[todss,pwr0,pwr1] = nt_dss0(C0,C1);
% look at power of the components (to pick which ones to keep)
figure; plot(pwr1./pwr0, '.-')
eye_comps = eegcat*todss(:,1:6);

% plot the components on a scalp
topo = [];
topo.component = 1:6;
topo.layout = lay;
ft_topoplotIC(topo,gt_ascomponent(eeg,todss));

% plot timecourse of the components
chans = cellfun(@(x)sprintf('eye%02d',x),num2cell(1:6),'UniformOutput',false);
this_plot = plot_cfg;
this_plot.ylim = [-1e-3 1e-3];
figure; ft_databrowser(this_plot, gt_asfieldtrip(eeg,eye_comps,'label',chans))

eegclean = nt_tsr(eegcat,eye_comps(:,4),time_shifts);

% rereference
eegreref = nt_rereference(eegclean,[outw(6:end-5,:) ones(size(eegclean,1),6)]);

eegfinal = gt_asfieldtrip(eeg,eegreref,'cropfirst',5,'croplast',5);
ft_databrowser(plot_cfg, eegfinal);

savename = regexprep(filename,'.bdf$','.eeg');
save_subject_binary(eegfinal,fullfile(data_dir,savename))

% subject 017 is complete garbage, there is no signal

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subject 018

filename = eegfiles(10).name;
filepath = fullfile(raw_data_dir,filename);
numstr = regexp(filepath,'([0-9]+)_','tokens');
sid = str2num(numstr{1}{1});

% read in the events
event_file = fullfile(data_dir,sprintf('sound_events_%03d.csv',sid));
stim_events = readtable(event_file);

% read in eeg data header
[eeg,ntrials] = gt_loadbdf(filepath,stim_events,'lengths',sound_lengths,...
    'channels',1:70,'padend',0.25);
% useful when we concatenate the trials for analysis

eeg = gt_downsample(eeg,stim_events,8);
eeg = gt_settrials(@nt_demean,eeg);
eeg = gt_settrials(@gt_detrend,eeg,[1 10],'progress','detrending...');
markers = cumsum(eeg.sampleinfo(:,2) - eeg.sampleinfo(:,1));

ft_databrowser(plot_cfg, eeg);

% remove two channels I know, immediately are bad
channels = [28];
eeg.hdr.label(channels) % shows the channels 'A28' and 'B25';
eeg = gt_settrials(@gt_interpolate_bad_channels,eeg,channels,closest,d,'channels',1:64);

freq = 0.5;
bad_indices = gt_fortrials(@nt_find_bad_channels,eeg,freq,3,150,1.5,'channels',1:64);
eeg.hdr.label(bad_indices{25}) % run this line to see which indices are bad for a given trial

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
% there are basically no glithes (after agressive channel imterpolation)
w = vertcat(w{:});
eegch = 1:64;
[outw,~] = gt_outliers(eegcat(:,eegch),w(:,eegch),5,4); % like nt_outliers, but shows a progress bar

% inspect the weights
this_plot = plot_cfg;
this_plot.ylim = [0 1];
ft_databrowser(this_plot,gt_asfieldtrip(eeg,[outw zeros(size(outw,1),6)]));
ft_databrowser(plot_cfg, eeg);

eegcat(:,eegch)=gt_inpaint(eegcat(:,eegch),outw); % interpolate over outliers

% visualize the data
ft_databrowser(plot_cfg, gt_asfieldtrip(eeg,eegcat));

% step 1: find regions of likely eye blinks and movement
eog = 67:70; % the sensors near the eyes
time_shifts = -5:5;
eyes = eegcat(:,eog);
[B,A]=butter(2,1/(eeg.hdr.Fs/2), 'high');
tmp = filter(B,A,eyes);
pcas=nt_pca(tmp,time_shifts,4);

% figure;
chans = cellfun(@(x)sprintf('pca%02d',x),num2cell(1:4),'UniformOutput',false);
this_plot = plot_cfg;
this_plot.ylim = [-30 30];
ft_databrowser(this_plot, gt_asfieldtrip(eeg,pcas,'label',chans,...
    'cropfirst',5,'croplast',5))

% compute signal envelope via hilbert transform
c = pcas(:,1);
c = abs(hilbert(c));
mask=abs(c)>5*median(abs(c));

eyemask = [eyes [mask; zeros(10,1)]*200];
chans = cellfun(@(x)sprintf('eye%02d',x),num2cell(1:4),'UniformOutput',false);
chans = [ chans 'mask' ];
this_plot = plot_cfg;
this_plot.ylim = [-200 200];
ft_databrowser(this_plot, gt_asfieldtrip(eeg,eyemask,'label',chans))

% step 2: find components using
C0=nt_cov(eegcat)
C1=nt_cov(bsxfun(@times,eegcat,[zeros(5,1);mask;zeros(5,1)]));
[todss,pwr0,pwr1] = nt_dss0(C0,C1);
% look at power of the components (to pick which ones to keep)
figure; plot(pwr1./pwr0, '.-')
comps = 1:12;
eye_comps = eegcat*todss(:,comps);

% plot the components on a scalp
topo = [];
topo.component = comps;
topo.layout = lay;
ft_topoplotIC(topo,gt_ascomponent(eeg,todss));

% plot timecourse of the components
chans = cellfun(@(x)sprintf('eye%02d',x),num2cell(comps),'UniformOutput',false);
this_plot = plot_cfg;
this_plot.ylim = [-1e-3 1e-3];
ft_databrowser(this_plot, gt_asfieldtrip(eeg,eye_comps,'label',chans))

eegclean = nt_tsr(eegcat,eye_comps(:,4),time_shifts);

% rereference
eegreref = nt_rereference(eegclean,[outw(6:end-5,:) ones(size(eegclean,1),6)]);

eegfinal = gt_asfieldtrip(eeg,eegreref,'cropfirst',5,'croplast',5);
ft_databrowser(plot_cfg, eegfinal);

savename = regexprep(filename,'.bdf$','.eeg');
save_subject_binary(eegfinal,fullfile(data_dir,savename))

}%
