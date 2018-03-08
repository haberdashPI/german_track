ft_defaults
analysisdir = 'models';
datadir = '/Users/davidlittle/Data/EEGAttn_David_Little_2018_01_24/';
eegfile = fullfile(datadir,'2018-01-24_0001_DavidLittle_biosemi.bdf');
modelfile_prefix = fullfile(analysisdir,...
                            '2018-01-24_0001_DavidLittle_target_model_1.0');

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
load(eegfile_proc);

% train the model
dat = load('config/experiment_record.mat');
all_stim_data = dat.experiment_cfg;
train_config = []
train_config.fs = all_stim_data.fs;
train_config.trial = {};
for trial = 1:length(eeg_data.trial)
  train_config.trial{trial} = ...
      get_stim_data(all_stim_data,stim_events(trial,:));
end

model_names = {'hit_target','miss_target'}
model = train_model(eeg_data,stim_events,train_config,...
                    model_names,modelfile_prefix,0);

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

grand_avg_trf = reduce_trf(@safeadd,model_names,model);
N = reduce_trf(@(x,y)x+1,model_names,0,model)

target_cor = stim_events;
target_cor{:,'target_time'} = NaN;
target_cor{:,'target'} = {'none'};
for i = 1:length(model_names)
  name = model_names{i};
  target_cor{:,[name '_cor']} = NaN;
end

all_cor_data = [];

% for each trial...
for trial = 1:height(stim_events)
  % if it exsits, do not include this trial's model
  trial_trf = map_trf(@cv_trf,model_names,...
                      grand_avg_trf,model{trial}.trf,N);

  target_cor{trial,'target_time'} = model{trial}.target_time;
  target_cor{trial,'target'} = {model{trial}.target};

  stim_config = train_config.trial{trial};

  for i = 1:length(model_names)
    name = model_names{i};

    if strcmp(name,'test_condition')
      stimcor = stim_config.stream.mixed;
    elseif strcmp(name,'object_condition')
      stimcor = stim_config.stream.male;
    elseif strcmp(name,'feature_condition')
      stimcor = stim_config.stream.right;
    elseif endsWith(name,'_target')
      if ~strcmp(model{trial}.target,'none')
        stimcor = stim_config.stream.(stim_config.target);
      end
    else
      error(['unrecognized condition "' name '".'])
    end

    if ~endsWith(name,'_target') || ~strcmp(model{trial}.target,'none')
      envelope = CreateLoudnessFeature(stimcor.data,stimcor.fs,...
                                       eeg_data.fsample);

      cor_data = ...
          model_cor_data(eeg_data.trial{trial},eeg_data.fsample,...
                          envelope,stimcor,model{trial},...
                          trial_trf.(name));

      cor = corrcoef(cor_data);
      target_cor{trial,[name '_cor']} = cor(1,2);

      all_cor_data(trial,i,:,:) = cor_data;
    end
  end
end

% TODO: get this right
all_cor_data(:,:,:,3) = repmat((1:size(all_cor_data,1))',...
                               [1 size(all_cor_data,2) size(all_cor_data,3)]);
all_cor_data(:,:,:,4) = repmat((1:size(all_cor_data,2)),...
                               [size(all_cor_data,1) 1 size(all_cor_data,3)]);

prediction = reshape(all_cor_data(:,:,:,1),[],1);
response = reshape(all_cor_data(:,:,:,2),[],1);
trial = reshape(all_cor_data(:,:,:,3),[],1);
model_index = reshape(all_cor_data(:,:,:,4),[],1);

table_cor_data = table(trial,response,prediction,model_index);

writetable(target_cor,'target_correlations_1.0.csv');
writetable(table_cor_data,'target_cor_data_1.0.csv');

% TODO:
% compute some aggregate correlations for each stream across time (i.e. using
% windows?)...
% do so for both misses and hits

% what we can be reasonably sure of: listeners were encoding the target
% near hits.
%
% based on the instruction we would expect listeners to attend
% to:
% the male stream during the object condition
% the right speaker during the feature condition
% the global mixture during the 'test' condition
% can we compare the correlations between these
% conditions to validate the model?
