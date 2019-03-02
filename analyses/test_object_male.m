run('../util/setup.m')

% First sanity check: can we more accurately recover the male voice
% when that is the voice listeners are asked to attend to.

dat = load(fullfile(data_dir,'config','experiment_record.mat'));
stim_info = dat.experiment_cfg;

eeg_files = dir(fullfile(data_dir,'eeg_response*.mat'));
[eeg,stim_events,sid] = load_subject(eeg_files(1).name);

maxlag = 0.25;
male_index = 1;
fs = stim_info.fs;
lags = 0:round(maxlag*eeg.fsample);

sum_model = []
for i = 1:length(eeg.trial)
  if strcmp(stim_events(i,'condition'),'object')
    sent_idx = stim_events{i,'sound_index'};
    stim = stim_info.all_sentences{1}{sent_idx(1),1};

    stim_envelope = CreateLoudnessFeature(stim,fs,eeg.fsample);
    response = eeg.trial{i};

    min_len = min(size(stim_envelope,1),size(response,2))
    response = response(:,1:min_len)';
    stim_envelope = stim_envelope(1:min_len);

    model = FindTRF(stim,response,-1,[],[],lags,'Shrinkage');
    if isempty(sum_model)
      sum_model = model
    else
      sum_model = sum_model + model
    end
  end
end
male_model = sum_model / 50;
