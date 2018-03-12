run('../util/setup.m')

modelfile_prefix = fullfile(model_dir,'target_model_2.0seconds');
results_prefix = fullfile(cache_dir,'target_2.0seconds');

% training configuration
dat = load(fullfile(data_dir,'config','experiment_record.mat'));
all_stim_data = dat.experiment_cfg;
train_config = [];
train_config.fs = all_stim_data.fs;
train_config.trial = {};

model_names = {'hit_target','miss_target'};

eeg_files = dir(fullfile(data_dir,'eeg_response*.bdf.mat'));

for file_index = 1:length(eeg_files)
  eventfile = fullfile(data_dir,sprintf('sound_events_%04d.csv',file_index));
  stim_events = readtable(eventfile);

  eegfile = fullfile(data_dir,eeg_files(file_index).name)
  eegfiledata = load(eegfile);
  eeg_data = eegfiledata.eeg_data;

  % train the model
  for trial = 1:length(eeg_data.trial)
    train_config.trial{trial} = ...
        get_stim_data(all_stim_data,stim_events(trial,:));
  end
  model = train_model(eeg_data,stim_events,train_config,...
                      model_names,sprintf('%s_sid%04d_',modelfile_prefix,file_index),1);

  % compute individual grand average
  grand_avg_trf = reduce_trf(@safeadd,model_names,model);
  N = reduce_trf(@(x,y)x+1,model_names,0,model)

  % setup data tables
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

    % compute results for each model type
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

  all_cor_data(:,:,:,3) = repmat((1:size(all_cor_data,1))',...
                                 [1 size(all_cor_data,2) size(all_cor_data,3)]);
  all_cor_data(:,:,:,4) = repmat((1:size(all_cor_data,2)),...
                                 [size(all_cor_data,1) 1 size(all_cor_data,3)]);

  prediction = reshape(all_cor_data(:,:,:,1),[],1);
  response = reshape(all_cor_data(:,:,:,2),[],1);
  trial = reshape(all_cor_data(:,:,:,3),[],1);
  model_index = reshape(all_cor_data(:,:,:,4),[],1);

  table_cor_data = table(trial,response,prediction,model_index);

  writetable(target_cor,sprintf('%s_sid%03d_cor.csv',results_prefix,file_index))
  writetable(table_cor_data,sprintf('%s_sid%03d_cor_data.csv',results_prefix,...
                                    file_index));
end

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
