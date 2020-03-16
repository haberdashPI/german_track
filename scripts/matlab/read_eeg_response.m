run(fullfile('..','..','src','matlab','util','setup.m'));
mkdir(fullfile(cache_dir,'eeg'));
usecache = 1; % whether to use previously preprocessed data stored in the cache

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

for i = 1:length(eegfiles)
    subject(i).load_channels = 1:70;
    subject(i).reref_first = false;
    subject(i).known_bad_channels = [];
    subject(i).bad_channel_threshs = {3,150,2};
    subject(i).eye_pca_comps = 1;
    subject(i).eye_mask_threshold = 4;
    subject(i).segment_outlier_thresh = 3;
end

subject(1).sid = 8;
subject(1).known_bad_channels = 28;

subject(2).sid = 9;
subject(2).load_channels = [1:64,129:134];
subject(2).known_bad_channels = 28;

subject(3).sid = 10;
subject(3).known_bad_channels = [28,57];
subject(3).eye_pca_comps = 2;

subject(4).sid = 11;
subject(4).known_bad_channels = 28;
subject(4).eye_pca_comps = 1;
subject(4).eye_mask_threshold = 3;

subject(5).sid = 12;
subject(5).known_bad_channels = 28;
subject(5).eye_pca_comps = 3;
subject(5).eye_mask_threshold = 4;

subject(6).sid = 13;
subject(6).known_bad_channels = 28;

subject(7).sid = 14;
subject(7).known_bad_channels = 28;

% subject 15 has no good data, file not generated

subject(8).sid = 16;
subject(8).known_bad_channels = 4;

subject(9).sid = 17;
subject(9).reref_first = true;
subject(9).known_bad_channels = 28;

subject(10).sid = 18;

subject(11).sid = 19;
subject(11).known_bad_channels = 57;

subject(12).sid = 20;
subject(12).sid = [];

subject(13).sid = 21;
subject(13).eye_mask_threshold = 3;

subject(14).sid = 22;
subject(14).reref_first = true;
subject(14).known_bad_channels = 28;

subject(15).sid = 23;
subject(15).sid = [];

subject(16).sid = 24;
subject(16).known_bad_channels = 28;

subject(17).sid = 25;

subject(18).sid = 26;
subject(18).sid = [];

subject(19).sid = 27;
subject(19).known_bad_channels = [22,28];

subject(20).sid = 28;
subject(20).known_bad_channels = [57,28];

subject(21).sid = 29;

subject(22).sid = 30;
subject(22).eye_mask_threshold = 3;

subject(23).sid = 31;
subject(23).known_bad_channels = [16,24,57,60,61];

subject(24).sid = 32;
subject(24).known_bad_channels = 63;

subject(25).sid = 33;
subject(25).known_bad_channels = 5;

subject(26).sid = 34;
subject(26).reref_first = true;
subject(26).known_bad_channels = 28;
subject(26).bad_channel_threshs = {2,150,1};
subject(26).eye_mask_threshold = 3;

subject(27).sid = 35;
subject(27).known_bad_channels = 28;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data = [];

% if you are rerunning analyses you can set interactive to false, if you are
% analyzing a new subject set this to true and run each section below,
% one at a time, using the plots to verify the results
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


    if isempty(subject(i).sid)
        warning("Subject id %d will be ignored.",sid);
        continue
    end
    savename = regexprep(filename,'.bdf$','.eeg');
    savetopath = fullfile(cache_dir,'eeg',savename);
    if isfile(savetopath)
        warning("Using cached subject data for sid %d.",sid)
        continue
    end

    %% read in the events
    event_file = fullfile(data_dir,sprintf('sound_events_%03d.csv',sid));
    stim_events = readtable(event_file);

    %% read in eeg data header
    eeg = gt_loadbdf(filepath,stim_events,'lengths',sound_lengths,'channels',subject(i).load_channels);
    if subject(i).reref_first
        eegcat = gt_fortrials(@(x)x,eeg);
        eegcat = vertcat(eegcat{:});
        eeg = gt_asfieldtrip(eeg,nt_rereference(eegcat));
    end

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
        eeg.hdr.label(bad_indices{1}) % run this line to see which indices are bad for a given trial
        this_plot = plot_cfg;
        this_plot.preproc.detrend = 'yes';
        ft_databrowser(plot_cfg, eeg);
    end

    %% interpolate bad channels
    eeg = gt_settrials(@gt_interpolate_bad_channels,{eeg,bad_indices},closest,dists,'channels',1:64);
    if interactive
        ft_databrowser(plot_cfg, eeg);
    end

    [weye,pcas] = gt_mask_eyeblinks(eeg,w,67:70,subject(i).eye_pca_comps,...
        subject(i).eye_mask_threshold);

    if interactive

        chans = cellfun(@(x)sprintf('pc%02d',x),num2cell(1:4),'UniformOutput',false);
        this_plot = plot_cfg;
        this_plot.ylim = [-400 400];
        ft_databrowser(this_plot, gt_asfieldtrip(eeg,pcas,'label',chans,'cropfirst',10));

        eyemask = min(horzcat(weye{:}))';
        eegcat = gt_fortrials(@(x)x,eeg);
        eegcat = vertcat(eegcat{:});
        chans = [ eeg.label; 'mask' ];
        ft_databrowser(plot_cfg, gt_asfieldtrip(eeg,[eegcat 20.*eyemask],'label',chans))

    end

    %% weighted rerefence
    eegcat = gt_fortrials(@(x)x,eeg);
    eegcat = vertcat(eegcat{:});
    eeg = gt_asfieldtrip(eeg,nt_rereference(eegcat));
    if interactive
        ft_databrowser(plot_cfg, eeg);
    end

    %% detect outlying segments
    [wseg,segnorm,segsd] = gt_segment_outliers(eeg,weye,...
        subject(i).segment_outlier_thresh);

    if interactive

        imagesc(segsd);

        plot(segnorm,'.-')
        mean(segnorm > subject(i).segment_outlier_thresh)

        segmask = min(horzcat(wseg{:}))';
        eegcat = gt_fortrials(@(x)x,eeg);
        eegcat = vertcat(eegcat{:});
        chans = [ eeg.label; 'mask' ];
        ft_databrowser(plot_cfg, gt_asfieldtrip(eeg,[eegcat 20.*segmask],'label',chans));

    end

    save_subject_binary(eeg,savetopath,'weights',wseg);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute correlation (C) of channel-subject features across all
% stimulus-condition pairs

cleaned_files = dir(fullfile(cache_dir,'eeg','*.eeg'));
maxlen = round(256*(max(sound_lengths)+0.5));
C = gt_mcca_C(cleaned_files,maxlen,{'global','object','spatial'},1:50,1:64);

[A,score,AA] = nt_mcca(C,n_chans,64);

bar(score(1:300));
