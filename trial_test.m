ft_defaults
analysisdir = 'models';
datadir = '/Users/davidlittle/Data/EEGAttn_David_Little_2018_01_24/';
eegfile = fullfile(datadir,'2018-01-24_0001_DavidLittle_biosemi.bdf');
modelfile_prefix = fullfile(analysisdir,'2018-01-24_0001_DavidLittle_model');

stim_events = readtable('sound_events.csv');
head = ft_read_header(eegfile);
fs = head.Fs;

baseline = 0;
trial_len = 10;
baseline_samples = floor(baseline*fs);
trial_len_samples = floor(trial_len*fs);

% TODO: turn this into a script that reads files
% from the original directoy and stores it in the data file
% (so we normally can just manage the preprocessed data).
% preprocessing
eegfile_proc = [eegfile '.proc.mat'];
if exist(eegfile_proc,'file') ~= 2
  % define the trials
  if exist(eegfile_proc,'file')
    error(['Cannot create a file with the same name as directory ' ...
           eegfile_proc]);
  end

  % load the trials
  cfg = [];
  cfg.dataset = eegfile;
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
  save(eegfile_proc,'eeg_data');
else
  load(eegfile_proc);
end

% train the model
model = train_model(eeg_data,stim_events,modelfile_prefix,1);

% questions
% do I use a broad or narrow time lag analysis?
% what is our ground truth? / santity check
%    we know that when people were correctly
%    reporting a target they were attending to
%    the speech target at that time,
%    and we know when they missed the target
%    that they weren't attending to it.

% if people are really using different strategies
% then during feature/object/global based analysis
% listener responses should be most accurately
% decoded by feature/object/global models.

% we can do a coarse time analyses, e.g.
% we can compare these three hypotehses during late and early
% targets.

N = length(model);
grand_avg_trf = reduce_trf(@(x,y) x+y,model)

target_cor = stim_events;
target_cor{:,'fem_young'} = NaN;
target_cor{:,'fem_old'} = NaN;
target_cor{:,'male'} = NaN;
target_cor{:,'target_time'} = NaN;
target_cor{:,'target'} = {'none'};

% for each trial...
for trial = 1:height(stim_events)
  cv_trial_trf = map_trf(@(grandm,trialm) (grandm - trialm)/(N-1),...
                             grand_avg_trf,model{trial}.trf);

  if ~strcmp(model{trial}.target,'none')
    target_cor{trial,'target_time'} = model{trial}.target_time;

    start = max(1,floor(eeg_data.fsample * (model{trial}.target_time - 1)));
    stop = min(ceil(eeg_data.fsample * (model{trial}.target_time + 0.5)),...
               size(eeg_data.trial{1},2));
    % start = 1;
    % stop = size(eeg_data.trial{1},2);

    names = {'fem_young','fem_old','male'};
    for i = 1:3
      target_cor{trial,names(i)} = ...
          model_correlate(start,stop,eeg_data.trial{trial},...
                          model{trial},cv_trial_trf,names{i});
    end
    target_cor{trial,'target'} = {model{trial}.target};

  end
end

writetable(target_cor,'target_correlations.csv');

% TODO:
% compute some aggregate correlations for each stream across time (i.e. using
% windows?)...
% do so for both misses and hits

% what we can be reasonably sure of: listeners were encoding the target
% near hits.
%
% TODO: train the model to the target near hits, and maybe (seperately?) to the
% two other sources near misses.
