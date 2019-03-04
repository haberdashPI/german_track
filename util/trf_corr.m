function C = trf_corr(eeg,stim_info,model,filter_fn,stim_fn)
  C = [];
  j = 0;
  for i = 1:length(eeg.trial)
    if filter_fn(i)
      stim = stim_fn(i);
      stim_envelope = CreateLoudnessFeature(stim,stim_info.fs,eeg.fsample);
      response = eeg.trial{i};

      min_len = min(size(stim_envelope,1),size(response,2));
      response = response(:,1:min_len)';
      stim_envelope = stim_envelope(1:min_len);

      [~,prediction] = FindTRF([],[],-1,response,model,lags,'Shrinkage');
      j = j+1;
      C(j) = corrcoef(prediction,stim_envelope);
    end
  end
end
