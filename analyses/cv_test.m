run('../util/setup.m')
usecache = 1;

dat = load(fullfile(data_dir,'config','experiment_record.mat'));
all_stim_data = dat.experiment_cfg;
eeg_files = dir(fullfile(data_dir,'eeg_response*_ica.bdf.mat'));

results_prefix = fullfile(cache_dir,'target_model_cca');

model_names = {'full_target_object','full_target_feature','full_target_global'};

modelfile_prefix = fullfile(model_dir,'full_target_model');
results_prefix = fullfile(cache_dir,'full_target_model');

method = 'CCA';
method_params = [];
method_params.regular = 1;

config.train = [];
config.train(1).model_name = 'full_target_feature';
config.train(1).maxlag = 0.25;
config.train(1).method = method;
config.train(1).method_params = method_params;
config.train(1).label = 'condition';
config.train(1).range = 'none';
config.train(1).filter = @(info,event) strcmp(event.condition,'feature');

config.train(2).model_name = 'full_target_object';
config.train(2).label = 'condition';
config.train(2).maxlag = 0.25;
config.train(2).method = method;
config.train(2).method_params = method_params;
config.train(2).range = 'none';
config.train(2).filter = @(info,event) strcmp(event.condition,'object');

config.train(3).model_name = 'full_target_global';
config.train(3).label = 'condition';
config.train(3).maxlag = 0.25;
config.train(3).method = method;
config.train(3).method_params = method_params;
config.train(3).range = 'none';
config.train(3).filter = @(info,event) strcmp(event.condition,'test');

config.test = [];
labels = {'target','mixture','non_targets'};
for i = 1:3
  config.test(i).models = {'full_target_object'};
  config.test(i).maxlag = 0.25;
  config.test(i).method = method;
  config.test(i).method_params = method_params;
  config.test(i).label = labels{i};
  config.test(i).range = [-2 0];
  config.test(i).filter = @(info,event) strcmp(event.condition,'object');
end

for i = 1:3
  config.test(i+3).models = {'full_target_feature'};
  config.test(i+3).maxlag = 0.25;
  config.test(i+3).method = method;
  config.test(i+3).method_params = method_params;
  config.test(i+3).label = labels{i};
  config.test(i+3).range = [-2 0];
  config.test(i+3).filter = @(info,event) strcmp(event.condition,'feature');
end

for i = 1:3
  config.test(i+6).models = {'full_target_global'};
  config.test(i+6).maxlag = 0.25;
  config.test(i+6).method = method;
  config.test(i+6).method_params = method_params;
  config.test(i+6).label = labels{i};
  config.test(i+6).range = [-2 0];
  config.test(i+6).filter = @(info,event) strcmp(event.condition,'test');
end

for sid_index = 1:length(eeg_files)
  [eeg_data,stim_events,sid] = load_trial(eeg_files(sid_index).name);

  all_cor = table();
  all_cor_data = table();



  textprogressbar('testing...')
  for trial = 1:length(eeg_data.trial)
    train_trials = setdiff(1:length(eeg_data.trial),trial);

    modelfile = sprintf('%s_sid%03d_trial%03d.mat',modelfile_prefix,sid,trial);
    if usecache && exist(modelfile,'file')
      mf = load(modelfile);
      model = mf.model;
    else
      model = train_model(eeg_data,all_stim_data,stim_events,config.train,...
                          train_trials);
      save(modelfile,'model')
    end

    for test_index = 1:length(config.test)
      test_config = config.test(test_index);
      audio = trial_audio(all_stim_data,stim_events(trial,:),test_config);

      textprogressbar(100*(trial-1) / length(eeg_data.trial));

      % run the test for each model the test applies to
      for model_index = 1:length(test_config.models)
        if ~isempty(audio.data)
          cor_data = ...
              test_model(eeg_data.trial{trial},eeg_data.fsample,...
                         audio,test_config,...
                         model.weights.(test_config.models{model_index}));

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

      textprogressbar(100*trial / length(eeg_data.trial));
    end

    writetable(all_cor,sprintf('%s_sid%03d_cor.csv',results_prefix,sid))
    writetable(all_cor_data,sprintf('%s_sid%03d_cor_data.csv',...
                                    results_prefix,sid));
  end
  textprogressbar('done!')
end
