run('../util/setup.m')

% First sanity check: can we more accurately recover the male voice
% when that is the voice listeners are asked to attend to.

eeg_files = dir(fullfile(data_dir,'eeg_response*.mat'));
[eeg,stim_events,sid] = load_subject(eeg_files(8).name);

male_index = 1

for i = 1:length(eeg.trial)
  if strcmp(events{i,'condition'},'object')
    lags = 0:round(maxlag*efs);
    [stim,fs] = trial_audio(info,stim_events(i))
    model = FindTRF(stim,response,-1,[],[],lags,'Shrinkage');
