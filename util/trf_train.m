function result = trf_train(eeg,stim_info,trial_filter,stim_fn)
  sum_model = [];
  n = 0;
  for i = 1:length(eeg.trial)
    if trial_filter(i)
      n = n+1;
      stim = stim_fn(i);
      stim_envelope = CreateLoudnessFeature(stim,stim_info.fs,eeg.fsample);
      response = eeg.trial{i};

      min_len = min(size(stim_envelope,1),size(response,2));
      response = response(:,1:min_len)';
      stim_envelope = stim_envelope(1:min_len);

      model = FindTRF(stim_envelope,response,-1,[],[],lags,'Shrinkage');
      if isempty(sum_model)
        sum_model = model;
      else
        sum_model = sum_model + model;
      end
    end
  end
  result = sum_model / n
end
