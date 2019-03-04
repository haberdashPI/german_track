function result = trf_train(eeg,stim_info,lags,filter_fn,stim_fn)
  sum_model = [];

  textprogressbar('training...');
  onCleanup(@() textprogressbar(''));
  N = sum(arrayfun(filter_fn,1:length(eeg.trial)));
  j = 0;
  for i = 1:length(eeg.trial)
    if filter_fn(i)
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

      j = j+1;
      textprogressbar(100*(j/N));
    end
  end
  result = sum_model / N;
end
