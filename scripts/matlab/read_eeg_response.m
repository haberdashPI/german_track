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
subject(1).reref_first = false;
subject(1).known_bad_channels = [];
subject(1).bad_channel_threshs = {3,150,2};

subject(2).sid = 9;
subject(2).load_channels = [1:64,129:134];
subject(2).reref_first = false;
subject(2).known_bad_channels = 28;
subject(2).bad_channel_threshs = {3,150,2};

subject(3).sid = 10;
subject(3).load_channels = 1:70;
subject(3).reref_first = false;
subject(3).known_bad_channels = [28,57];
subject(3).bad_channel_threshs = {3,150,2};
subject(3).fix_glitches = false;

subject(4).sid = 11;
subject(4).load_channels = 1:70;
subject(4).reref_first = false;
subject(4).known_bad_channels = [28];
subject(4).bad_channel_threshs = {3,150,2};

subject(5).sid = 12;
subject(5).load_channels = 1:70;
subject(5).reref_first = false;
subject(5).known_bad_channels = [28];
subject(5).bad_channel_threshs = {3,150,2};

subject(6).sid = 13;
subject(6).load_channels = 1:70;
subject(6).reref_first = false;
subject(6).known_bad_channels = [28];
subject(6).bad_channel_threshs = {3,150,2};
subject(6).fix_glitches = true;

subject(7).sid = 14;
subject(7).reref_first = false;
subject(7).load_channels = 1:70;
subject(7).known_bad_channels = [28];
subject(7).bad_channel_threshs = {3,150,2};

subject(8).sid = 16;
subject(8).reref_first = false;
subject(8).load_channels = 1:70;
subject(8).known_bad_channels = [4];
subject(8).bad_channel_threshs = {3,150,2};

subject(9).sid = 17;
subject(9).reref_first = true;
subject(9).load_channels = 1:70;
subject(9).known_bad_channels = [28];
subject(9).bad_channel_threshs = {3,150,2};

subject(10).sid = 18;
subject(10).reref_first = false;
subject(10).load_channels = 1:70;
subject(10).known_bad_channels = [];
subject(10).bad_channel_threshs = {3,150,2};

subject(11).sid = 19;
subject(11).reref_first = false;
subject(11).load_channels = 1:70;
subject(11).known_bad_channels = 57;
subject(11).bad_channel_threshs = {3,150,2};

subject(12).sid = [];

subject(13).sid = 21;
subject(13).reref_first = false;
subject(13).load_channels = 1:70;
subject(13).known_bad_channels = [];
subject(13).bad_channel_threshs = {3,150,2};

subject(14).sid = 22;
subject(14).reref_first = true;
subject(14).load_channels = 1:70;
subject(14).known_bad_channels = 28;
subject(14).bad_channel_threshs = {2,150,0.75};

subject(15).sid = [];

subject(16).sid = 24;
subject(16).reref_first = false;
subject(16).load_channels = 1:70;
subject(16).known_bad_channels = 28;
subject(16).bad_channel_threshs = {3,150,2};

subject(17).sid = 25;
subject(17).reref_first = false;
subject(17).load_channels = 1:70;
subject(17).known_bad_channels = [];
subject(17).bad_channel_threshs = {3,150,2};

subject(18).sid = [];

subject(19).sid = 27;
subject(19).reref_first = false;
subject(19).load_channels = 1:70;
subject(19).known_bad_channels = [22,28];
subject(19).bad_channel_threshs = {3,150,2};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data = [];

% if you are rerunning analyses you can set this to false, if you are analyzing a
% new subject set this to true and run each section below, one at a time, using
% the plots to verify the results
interactive = false;
for i = 5:7 %1:7 %length(eegfiles)

    %% file information
    filename = eegfiles(i).name;
    filepath = fullfile(raw_data_dir,filename);
    numstr = regexp(filepath,'([0-9]+)_','tokens');
    sid = str2num(numstr{1}{1});
    if isempty(subject(i).sid)
        warning("Subject id %d will be ignored.",sid);
    end
    if subject(i).sid ~= sid
        error('Wrong subject id (%d) specified for index %d, expected sid %d',...
            subject(i).sid,i,sid);
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

    %% rerefence
    eegcat = gt_fortrials(@(x)x,eeg);
    eegcat = vertcat(eegcat{:});
    wcat = vertcat(w{:});
    eegreref = gt_asfieldtrip(eeg,nt_rereference(eegcat,wcat));
    if interactive
        ft_databrowser(plot_cfg, eegreref);
    end

    data(i).eeg = eegreref;
    data(i).w = w;
end
