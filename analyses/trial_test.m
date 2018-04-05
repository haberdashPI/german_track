run('../util/setup.m')

usecache = 0;

% TODO: these configurations need to use the new interface
% modelfile_prefix = fullfile(model_dir,'condition_model');
% results_prefix = fullfile(cache_dir,'condition_model');
% model_names = {'test_condition','object_condition','feature_condition'};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_names = {'hit_target','miss_mixture','miss_nontargets'}
modelfile_prefix = fullfile(model_dir,'target_model');
results_prefix = fullfile(cache_dir,'target_model');

config.train = [];
config.train(1).model_name = 'hit_target';
config.train(1).label = 'target';
config.train(1).range = [-2 0];
config.train(1).filter = @(info,event) strcmp(event.response{:},'2');

config.train(2).model_name = 'miss_mixture';
config.train(2).label = 'mixture';
config.train(2).range = [-2 0];
config.train(2).filter = @(info,event) strcmp(event.response{:},'3');

config.train(3).model_name = 'miss_nontargets';
config.train(3).label = 'non_targets';
config.train(3).range = [-2 0];
config.train(3).filter = @(info,event) strcmp(event.response{:},'3');

config.test = [];
config.test(1).models = {'hit_target'};
config.test(1).label = 'target';
config.test(1).range = [-2 0];
config.test(1).filter = @(info,event) 1;

config.test(2).models = {'miss_mixture'};
config.test(2).label = 'mixture';
config.test(2).range = [-2 0];
config.test(2).filter = @(info,event) 1;

config.test(3).models = {'miss_nontargets'};
config.test(3).label = 'non_targets';
config.test(3).range = [-2 0];
config.test(3).filter = @(info,event) 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% model_names = {'general_envelope'};

% modelfile_prefix = fullfile(model_dir,'genenv_model');
% results_prefix = fullfile(cache_dir,'genenv_model');

% config.train = [];
% config.train(1).model_name = 'general_envelope';
% config.train(1).label = 'condition';
% config.train(1).range = 'none';
% config.train(1).filter = @(info,event) 1;

% config.test = [];
% config.test(1).models = {'general_envelope'};
% config.test(1).label = 'target';
% config.test(1).range = [-2 0];
% config.test(1).filter = @(info,event) 1;

% config.test(2).models = {'general_envelope'};
% config.test(2).label = 'mixture';
% config.test(2).range = [-2 0];
% config.test(2).filter = @(info,event) 1;

% config.test(3).models = {'general_envelope'};
% config.test(3).label = 'non_targets';
% config.test(3).range = [-2 0];
% config.test(3).filter = @(info,event) 1;

% config.test(4).models = {'general_envelope'};
% config.test(4).label = 'non_target1';
% config.test(4).range = [-2 0];
% config.test(4).filter = @(info,event) 1;

% config.test(5).models = {'general_envelope'};
% config.test(5).label = 'non_target2';
% config.test(5).range = [-2 0];
% config.test(5).filter = @(info,event) 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% model_names = {'full_target_object','full_target_feature','full_target_global'};

% modelfile_prefix = fullfile(model_dir,'full_target_model');
% results_prefix = fullfile(cache_dir,'full_target_model');

% config.train = [];
% config.train(1).model_name = 'full_target_feature';
% config.train(1).label = 'condition';
% config.train(1).range = 'none';
% config.train(1).filter = @(info,event) strcmp(event.condition,'feature');

% config.train(2).model_name = 'full_target_object';
% config.train(2).label = 'condition';
% config.train(2).range = 'none';
% config.train(2).filter = @(info,event) strcmp(event.condition,'object');

% config.train(3).model_name = 'full_target_global';
% config.train(3).label = 'condition';
% config.train(3).range = 'none';
% config.train(3).filter = @(info,event) strcmp(event.condition,'test');

% config.test = [];
% labels = {'target','mixture','non_targets'};
% for i = 1:3
%   config.test(i).models = {'full_target_object'};
%   config.test(i).label = labels{i};
%   config.test(i).range = [-2 0];
%   config.test(i).filter = @(info,event) strcmp(event.condition,'object');
% end

% for i = 1:3
%   config.test(i+3).models = {'full_target_feature'};
%   config.test(i+3).label = labels{i};
%   config.test(i+3).range = [-2 0];
%   config.test(i+3).filter = @(info,event) strcmp(event.condition,'feature');
% end

% for i = 1:3
%   config.test(i+6).models = {'full_target_global'};
%   config.test(i+6).label = labels{i};
%   config.test(i+6).range = [-2 0];
%   config.test(i+6).filter = @(info,event) strcmp(event.condition,'test');
% end

% training configuration
dat = load(fullfile(data_dir,'config','experiment_record.mat'));
all_stim_data = dat.experiment_cfg;

eeg_files = dir(fullfile(data_dir,'eeg_response*.bdf.mat'));

for sid_index = 2 %sid_index = 1:length(eeg_files)
  eegfile = fullfile(data_dir,eeg_files(sid_index).name)
  eegfiledata = load(eegfile);
  eeg_data = eegfiledata.eeg_data;

  numstr = regexp(eegfile,'_([0-9]+).bdf','tokens');
  sid = str2num(numstr{1}{1});
  eventfile = fullfile(data_dir,sprintf('sound_events_%03d.csv',sid));
  stim_events = readtable(eventfile);

  % train the model
  model = train_model(eeg_data,all_stim_data,stim_events,config.train,...
                      sprintf('%s_sid%03d',modelfile_prefix,sid),...
                      usecache);

  % compute individual grand average
  grand_avg_trf = reduce_trf(@safeadd,model_names,model);
  N = reduce_trf(@(x,y)x+1,model_names,0,model);

  % setup data tables
  all_cor = table();
  all_cor_data = table();
  % all_cor_data = zeros(height(stim_events),length(model_names),...
  %                      size(eeg_data.trial{1},2))*NaN;

  % for each trial...
  textprogressbar('computing correlations: ');
  try
    for trial = 1:height(stim_events)
      % if it exsits, do not include this trial's model
      trial_trf = map_trf(@cv_trf,model_names,...
                          grand_avg_trf,model{trial}.trf,N);

      % run each test
      for test_index = 1:length(config.test)
        stim = config.test(test_index);
        audio = trial_audio(all_stim_data,stim_events(trial,:),stim);

        % run the test for each model the test applies to
        for model_index = 1:length(stim.models)
          if ~isempty(audio.data)
            envelope = CreateLoudnessFeature(audio.data,audio.fs,eeg_data.fsample);

            cor_data = ...
                model_cor_data(eeg_data.trial{trial},eeg_data.fsample,...
                               envelope,audio,model{trial},...
                               trial_trf.(stim.models{model_index}));

            cor = corrcoef(cor_data);
            name = sprintf('%s_to_%s_cor',stim.models{model_index},stim.label);

            row = stim_events(trial,:);
            row(:,'cor') = {cor(1,2)};
            row(:,'sid') = {sid};
            row(:,'model') = stim.models(model_index);
            row(:,'test') = {stim.label};
            row(:,'target_time') = {audio.target_time};
            all_cor = [all_cor; row];

            prediction = cor_data(:,1);
            response = cor_data(:,2);
            cor_data = table(prediction,response);
            cor_data(:,'cor') = {cor(1,2)};
            cor_data(:,'sid') = {sid};
            cor_data(:,'model') = stim.models(model_index);
            cor_data(:,'test') = {stim.label};

            all_cor_data = [all_cor_data; cor_data];
          end
        end
      end
      textprogressbar(100*(trial / height(stim_events)));
    end
    textprogressbar('finished!');
  catch e
    textprogressbar('error!');
    rethrow(e);
  end

  writetable(all_cor,sprintf('%s_sid%03d_cor.csv',results_prefix,sid))
  writetable(all_cor_data,sprintf('%s_sid%03d_cor_data.csv',results_prefix,sid));
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
