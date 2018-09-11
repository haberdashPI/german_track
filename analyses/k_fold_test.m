run('../util/setup.m')
usecache = 1;

dat = load(fullfile(data_dir,'config','experiment_record.mat'));
all_stim_data = dat.experiment_cfg;
eeg_files = dir(fullfile(data_dir,'eeg_response*_ica.bdf.mat'));
results_prefix = fullfile(cache_dir,'envelope_cca');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% configuartion
K = 10;
model_names = {'cca_feature','cca_object','cca_global'};

% TODO: rename to include "fake" in name
noise = 0;
shared_prefix = 'fake_0';
modelfile_prefix = fullfile(model_dir,shared_prefix);
results_prefix = fullfile(cache_dir,shared_prefix);

method = 'CCA_envelope';
method_params = [];
method_params.regular = 1;

warning('Using fake data to train and test the model!')

config.train = [];
labels = {'object','feature','test'};
for i = 1:3
  config.train(i).model_name = ['cca_' labels{i}];
  config.train(i).maxlag = 0.25;
  config.train(i).method = method;
  config.train(i).method_params = method_params;
  config.train(i).label = labels;
  config.train(i).range = 'none';
  config.train(i).filter = @(info,event) strcmp(event.condition,labels{i}) ...
    && has_target(info,event);

  config.train(i).fake_seed = @(trial) trial;
  config.train(i).fake_data = 1;
  config.train(i).fake_channels = {1:10, 1:10, 1:10};
  config.train(i).fake_channels{i} = 1:50;
  config.train(i).fake_lag = 3;
  config.train(i).fake_noise = noise;
end

config.test = [];
labels = {'object','feature','test'};
for i = 1:3
  config.test(i).models = {['cca_' labels{i}]};
  config.test(i).maxlag = 0.25;
  config.test(i).method = method;
  config.test(i).method_params = method_params;
  config.test(i).label = labels;
  config.test(i).range = [-2 0];
  config.test(i).filter = @(info,event) strcmp(event.condition,labels{i}) ...
    && has_target(info,event);

  config.test(i).fake_seed = @(trial) trial+10000;
  config.test(i).fake_data = 1;
  config.test(i).fake_channels = {1:10, 1:10, 1:10};
  config.test(i).fake_channels{i} = 1:50;
  config.test(i).fake_lag = 3;
  config.test(i).fake_noise = noise;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for sid_index = 2 %1:length(eeg_files)
  [eeg_data,stim_events,sid] = load_trial(eeg_files(sid_index).name);

  all_cor = table();
  all_cor_data = table();

  textprogressbar('testing...')
  fold_size = ceil(length(eeg_data.trial) / K);
  for fold = 1:K
    test_trials = (fold-1)*fold_size+1 : ...
      min(length(eeg_data.trial),fold*fold_size);

    train_trials = setdiff(1:length(eeg_data.trial),test_trials);

    modelfile = sprintf('%s_sid%03d_fold%02d.mat',modelfile_prefix,sid,fold);
    if usecache && exist(modelfile)
      mf = load(modelfile);
      model = mf.model;
    else
      model = train_model(eeg_data,all_stim_data,stim_events,config.train,...
                          train_trials);
      save(modelfile,'model');
    end

    for trial = test_trials
      % disp(['Trial: ' num2str(trial)])
      for test_index = 1:length(config.test)
        test_config = config.test(test_index);
        textprogressbar(100*(trial-1) / length(eeg_data.trial));

        % run the test for each model the test applies to
        for model_index = 1:length(test_config.models)
          if test_config.filter(all_stim_data,stim_events(trial,:))
            [stim_result,eeg_result] = ...
              prepare_data(all_stim_data,test_config,stim_events(trial,:),...
                           eeg_data,trial);
            weights = model.weights.(test_config.models{model_index});
            cca_stim = eeg_result * weights.eeg;

            row = stim_events(trial,:);
            row(:,'sid') = {sid};
            row(:,'model') = test_config.models(model_index);
            row(:,'target_time') = {get_target_time(all_stim_data,...
              stim_events(trial,:))};

            labels = {'object','feature','test'};
            for i = 1:3
              cor = corrcoef([cca_stim(:,i) stim_result(:,i)]);
              row(:,['cor_' labels{i}]) = {cor(1,2)};
            end

            all_cor = [all_cor; row];
          end
        end

        textprogressbar(100*trial / length(eeg_data.trial));
      end
    end
    writetable(all_cor,sprintf('%s_sid%03d_cor.csv',results_prefix,sid));
    writetable(all_cor_data,sprintf('%s_sid%03d_cor_data.csv',...
      results_prefix,sid));
  end
  textprogressbar('done!');
end
