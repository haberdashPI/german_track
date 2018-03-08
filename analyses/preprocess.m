run('../util/setup.m')

eegfiles = dir(fullfile(raw_data_dir,'*.bdf'))
for i = 1:length(eegfiles)
  eegfile = eegfiles(i).name;
  numstr = regexp(eegfile,'_([0-9]+)_','tokens');
  sid = str2num(numstr{1}{1});
  result_file = fullfile(data_dir,sprintf('eeg_response_%04d.bdf.mat',sid));

  if exist(result_file)
    warning(['The file ' result_file ' already exists. Skipping...']);
    continue;
  end

  disp(['reading events for ' eegfile]);
  disp(['Found SID = ' num2str(sid)]);


  event_file = fullfile(data_dir,sprintf('sound_events_%04d.csv',sid));
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
  save(result_file,'eeg_data');
end
