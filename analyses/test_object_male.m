run('../util/setup.m')

% First sanity check: can we more accurately recover the male voice
% when that is the voice listeners are asked to attend to.

dat = load(fullfile(data_dir,'config','experiment_record.mat'));
stim_info = dat.experiment_cfg;

eeg_files = dir(fullfile(data_dir,'eeg_response*.mat'));
[eeg,stim_events,sid] = load_subject(eeg_files(1).name);

maxlag = 0.25;

male_index = 1;
fem1_index = 2;
fem2_index = 2;

fs = stim_info.fs;
lags = 0:round(maxlag*eeg.fsample);

male_model = trf_train(eeg,stim_info,...
  @(i)strcmp(stim_events(i,'condition'),'object'),...
  @(i)sentence(stim_events,stim_info,i,male_index));

fem1_model = trf_train(eeg,stim_info,...
  @(i)strcmp(stim_events(i,'condition'),'object'),...
  @(i)sentence(stim_events,stim_info,i,fem1_index));

fem2_model = trf_train(eeg,stim_info,...
  @(i)strcmp(stim_events(i,'condition'),'object'),...
  @(i)sentence(stim_events,stim_info,i,fem2_index));


% TODO: once I've written this up, make it a function
for i = 1:length(eeg.trial)
  if strcmp(stim_events(i,'condition'),'object')
    [~,prediction] = FindTRF([],[],-1,eeg.trial{1}',male_model,...
        lags,'Shrinkage');

    % TODO: compute the envelope
    % TODO: create cor array

    j = j+1
    cor(j) = corrcoef(prediction,envelope);
  end
end
