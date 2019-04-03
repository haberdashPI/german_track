run('../util/setup.m')
usecache = 1;

eegfiles = dir(fullfile(raw_data_dir,'*.bdf'));
for i = 1:length(eegfiles)
    eegfile = eegfiles(i).name;
    numstr = regexp(eegfile,'([0-9]+)_','tokens');
    sid = str2num(numstr{1}{1});
    result_file = fullfile(data_dir,sprintf('eeg_response_%03d.mat',sid));

    if exist(result_file,'file') && usecache
        warning(['The file ' result_file ' already exists. Skipping...']);
        continue;
    end

    disp(['reading responses for ' eegfile]);
    disp(['Found SID = ' num2str(sid)]);

    event_file = fullfile(data_dir,sprintf('sound_events_%03d.csv',sid));
    stim_events = readtable(event_file);
    head = ft_read_header(fullfile(raw_data_dir,eegfile));
    fs = head.Fs;

    baseline = 0;
    trial_len = 10;
    baseline_samples = floor(baseline*fs);
    trial_len_samples = floor(trial_len*fs);

    % load the trials
    cfg = [];
    cfg.dataset = fullfile(raw_data_dir,eegfile);
    cfg.trl = [max(0,stim_events{:,'sample'}-baseline_samples) ...
    min(head.nSamples,stim_events{:,'sample'}+trial_len_samples) ...
    baseline_samples*ones(height(stim_events),1)];
    cfg.continuous = 'yes';
    if sid == 1
        cfg.channel = [1:128 257:264];
    elseif sid == 9
        cfg.channel = 1:136;
    else
        cfg.channel = 1:72;
    end

    % apply a (1Hz) high pass filter
    % (a recent study suggests that maybe I should remove this:
    % need to examine either replacing with robust filter or just doing nothing)
    cfg.hpfilter = 'yes';
    cfg.hpfreq = 1;
    cfg.hpfiltortype = 'but';
    cfg.hpfiltord = 4;

    eeg_data = ft_preprocessing(cfg);

    % downsample the trials to 64Hz
    cfg = [];
    cfg.resamplefs = 64;
    eeg_data = ft_resampledata(cfg,eeg_data);

    % re-reference the data
    cfg = [];
    cfg.refchannel = 'all';
    cfg.reref = 'yes';
    eeg_data = ft_preprocessing(cfg,eeg_data);

    % save to a file
    ft_write_data(result_file,eeg_data,'dataformat','matlab');
end
alert()
