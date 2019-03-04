run('../util/setup.m')

% First sanity check: can we more accurately recover the male voice
% when that is the voice listeners are asked to attend to.

dat = load(fullfile(data_dir,'config','experiment_record.mat'));
stim_info = dat.experiment_cfg;

eeg_files = dir(fullfile(data_dir,'eeg_response*.mat'));

male_C = [];
fem1_C = [];
fem2_C = [];
sid = {};
for i = 1:length(eeg_files)
  [eeg,stim_events,cur_sid] = load_subject(eeg_files(i).name);

  maxlag = 0.25;

  male_index = 1;
  fem1_index = 2;
  fem2_index = 3;

  lags = 0:round(maxlag*eeg.fsample);

  male_model = trf_train(eeg,stim_info,lags,...
    @(i)strcmp(stim_events{i,'condition'},'object'),...
    @(i)sentence(stim_events,stim_info,i,male_index));

  fem1_model = trf_train(eeg,stim_info,lags,...
    @(i)strcmp(stim_events{i,'condition'},'object'),...
    @(i)sentence(stim_events,stim_info,i,fem1_index));

  fem2_model = trf_train(eeg,stim_info,lags,...
    @(i)strcmp(stim_events{i,'condition'},'object'),...
    @(i)sentence(stim_events,stim_info,i,fem2_index));


  this_male_C = trf_corr(eeg,stim_info,male_model,lags,...
    @(i)strcmp(stim_events{i,'condition'},'object'),...
    @(i)sentence(stim_events,stim_info,i,male_index))';

  this_fem1_C = trf_corr(eeg,stim_info,fem1_model,lags,...
    @(i)strcmp(stim_events{i,'condition'},'object'),...
    @(i)sentence(stim_events,stim_info,i,fem1_index))';

  this_fem2_C = trf_corr(eeg,stim_info,fem2_model,lags,...
    @(i)strcmp(stim_events{i,'condition'},'object'),...
    @(i)sentence(stim_events,stim_info,i,fem2_index))';

  cur_sids = cell(length(this_fem1_C),1);
  cur_sids(:) = {cur_sid};

  male_C = [male_C; this_male_C];
  fem1_C = [fem1_C; this_fem1_C];
  fem2_C = [fem2_C; this_fem2_C];
  sid = [sid; cur_sids];
end

writetable(table(male_C,fem1_C,fem2_C,sid),fullfile(cache_dir,'testobj.csv'));
