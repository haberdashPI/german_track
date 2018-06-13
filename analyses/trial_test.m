run('../util/setup.m')

usecache = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_names = {'hit_target','miss_mixture','miss_nontargets'};
modelfile_prefix = fullfile(model_dir,'target_model_cca');
results_prefix = fullfile(cache_dir,'target_model_cca');

method = 'CCA'
method_params = [];
method_params.regular = 1;

% loadload;

config.train = [];
config.train(1).model_name = 'hit_target';
config.train(1).maxlag = 0.25;
config.train(1).method = method;
config.train(1).method_params = method_params;
config.train(1).label = 'target';
config.train(1).range = [-2 0];
config.train(1).filter = @(info,event) strcmp(event.response{:},'2');

config.train(2).model_name = 'miss_mixture';
config.train(2).label = 'mixture';
config.train(2).maxlag = 0.25;
config.train(2).method = method;
config.train(2).method_params = method_params;
config.train(2).range = [-2 0];
config.train(2).filter = @(info,event) strcmp(event.response{:},'3');

config.train(3).model_name = 'miss_nontargets';
config.train(3).label = 'non_targets';
config.train(3).maxlag = 0.25;
config.train(3).method = method;
config.train(3).method_params = method_params;
config.train(3).range = [-2 0];
config.train(3).filter = @(info,event) strcmp(event.response{:},'3');

config.test = [];
config.test(1).models = {'hit_target'};
config.test(1).maxlag = 0.25;
config.test(1).method = method;
config.test(1).method_params = method_params;
config.test(1).label = 'target';
config.test(1).range = [-2 0];
config.test(1).filter = @(info,event) 1;

config.test(2).models = {'miss_mixture'};
config.test(2).label = 'mixture';
config.test(2).maxlag = 0.25;
config.test(2).method = method;
config.test(2).method_params = method_params;
config.test(2).range = [-2 0];
config.test(2).filter = @(info,event) 1;

config.test(3).models = {'miss_nontargets'};
config.test(3).label = 'non_targets';
config.test(3).maxlag = 0.25;
config.test(3).method = method;
config.test(3).method_params = method_params;
config.test(3).range = [-2 0];
config.test(3).filter = @(info,event) 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% model_names = {'general_envelope'};

% modelfile_prefix = fullfile(model_dir,'genenv_model');
% results_prefix = fullfile(cache_dir,'genenv_model');

% config.train = [];
% config.train(1).model_name = 'general_envelope';
% config.train(1).maxlag = 0.25;
% config.train(1).method = 'TRF';
% config.train(1).label = 'condition';
% config.train(1).range = 'none';
% config.train(1).filter = @(info,event) 1;

% config.test = [];
% config.test(1).models = {'general_envelope'};
% config.test(1).maxlag = 0.25;
% config.test(1).method = 'TRF';
% config.test(1).label = 'target';
% config.test(1).range = [-2 0];
% config.test(1).filter = @(info,event) 1;

% config.test(2).models = {'general_envelope'};
% config.test(2).label = 'mixture';
% config.test(2).maxlag = 0.25;
% config.test(2).method = 'TRF';
% config.test(2).range = [-2 0];
% config.test(2).filter = @(info,event) 1;

% config.test(3).models = {'general_envelope'};
% config.test(3).label = 'non_targets';
% config.test(3).maxlag = 0.25;
% config.test(3).method = 'TRF';
% config.test(3).range = [-2 0];
% config.test(3).filter = @(info,event) 1;

% config.test(4).models = {'general_envelope'};
% config.test(4).label = 'non_target1';
% config.test(4).maxlag = 0.25;
% config.test(4).method = 'TRF';
% config.test(4).range = [-2 0];
% config.test(4).filter = @(info,event) 1;

% config.test(5).models = {'general_envelope'};
% config.test(5).label = 'non_target2';
% config.test(5).maxlag = 0.25;
% config.test(5).method = 'TRF';
% config.test(5).range = [-2 0];
% config.test(5).filter = @(info,event) 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% model_names = {'full_target_object','full_target_feature','full_target_global'};

% modelfile_prefix = fullfile(model_dir,'full_target_model');
% results_prefix = fullfile(cache_dir,'full_target_model');

% config.train = [];
% config.train(1).model_name = 'full_target_feature';
% config.train(1).maxlag = 0.25;
% config.train(1).method = 'TRF';
% config.train(1).label = 'condition';
% config.train(1).range = 'none';
% config.train(1).filter = @(info,event) strcmp(event.condition,'feature');

% config.train(2).model_name = 'full_target_object';
% config.train(2).label = 'condition';
% config.train(2).maxlag = 0.25;
% config.train(2).method = 'TRF';
% config.train(2).range = 'none';
% config.train(2).filter = @(info,event) strcmp(event.condition,'object');

% config.train(3).model_name = 'full_target_global';
% config.train(3).label = 'condition';
% config.train(3).maxlag = 0.25;
% config.train(3).method = 'TRF';
% config.train(3).range = 'none';
% config.train(3).filter = @(info,event) strcmp(event.condition,'test');

% config.test = [];
% labels = {'target','mixture','non_targets'};
% for i = 1:3
%   config.test(i).models = {'full_target_object'};
%   config.test(i).maxlag = 0.25;
%   config.test(i).method = 'TRF';
%   config.test(i).label = labels{i};
%   config.test(i).range = [-2 0];
%   config.test(i).filter = @(info,event) strcmp(event.condition,'object');
% end

% for i = 1:3
%   config.test(i+3).models = {'full_target_feature'};
%   config.test(i+3).maxlag = 0.25;
%   config.test(i+3).method = 'TRF';
%   config.test(i+3).label = labels{i};
%   config.test(i+3).range = [-2 0];
%   config.test(i+3).filter = @(info,event) strcmp(event.condition,'feature');
% end

% for i = 1:3
%   config.test(i+6).models = {'full_target_global'};
%   config.test(i+6).maxlag = 0.25;
%   config.test(i+6).method = 'TRF';
%   config.test(i+6).label = labels{i};
%   config.test(i+6).range = [-2 0];
%   config.test(i+6).filter = @(info,event) strcmp(event.condition,'test');
% end

% training configuration
dat = load(fullfile(data_dir,'config','experiment_record.mat'));
all_stim_data = dat.experiment_cfg;

eeg_files = dir(fullfile(data_dir,'eeg_response*_ica.bdf.mat'));

for sid_index = 1:length(eeg_files)
  [eeg_data,stim_events,sid] = load_trial(eeg_files(sid_index).name)

  % train the model
  model = {};
  for trial = 1:length(eeg.trial)
      modelfile = sprintf('%s_sid%03d_trial%03d.mat',modelfile_prefix,sid,trial);
      if usecache && exist(modelfile)
        mf = load(modelfile)
        model{trial} = mf.model_trial
      else
        model_trial = train_model(eeg_data,all_stim_data,stim_events,...
                                  config.trian,[trial])
        save(modelfile,'model_trial')
        model{trial} = model_trial
      end
  end

  % compute individual grand average
  grand_avg_weights = reduce_weights(@safeadd,model_names,model);
  N = reduce_weights(@(x,y)x+1,model_names,0,model);

  % setup data tables
  all_cor = table();
  all_cor_data = table();
  % all_cor_data = zeros(height(stim_events),length(model_names),...
  %                      size(eeg_data.trial{1},2))*NaN;

  % for each trial...
  textprogressbar('computing correlations: ');
  try
    for trial = 1:length(eeg_data.trial)
      % if it exsits, do not include this trial's model
      trial_weights = map_weights(@cv_weights,model_names,...
                                  grand_avg_weights,model{trial}.weights,N);

      % run each test
      for test_index = 1:length(config.test)
        test_config = config.test(test_index);
        audio = trial_audio(all_stim_data,stim_events(trial,:),test_config);

        % run the test for each model the test applies to
        for model_index = 1:length(test_config.models)
          if ~isempty(audio.data)
            cor_data = ...
                test_model(eeg_data.trial{trial},eeg_data.fsample,...
                           audio,test_config,...
                           trial_weights.(test_config.models{model_index}));

            cor = corrcoef(cor_data);
            name = sprintf('%s_to_%s_cor',test_config.models{model_index},...
                           test_config.label);

            row = stim_events(trial,:);
            row(:,'cor') = {cor(1,2)};
            row(:,'sid') = {sid};
            row(:,'model') = test_config.models(model_index);
            row(:,'test') = {test_config.label};
            row(:,'target_time') = {audio.target_time};
            all_cor = [all_cor; row];

            prediction = cor_data(:,1);
            response = cor_data(:,2);
            cor_data = table(prediction,response);
            cor_data(:,'cor') = {cor(1,2)};
            cor_data(:,'sid') = {sid};
            cor_data(:,'model') = test_config.models(model_index);
            cor_data(:,'test') = {test_config.label};

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
