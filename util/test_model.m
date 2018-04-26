function cor_data = test_model(eeg_data,efs,audio,config,weights)
  start = max(1,floor(efs * audio.start));
  stop = min(ceil(efs * audio.stop),size(eeg_data,2));

  lags = 0:round(config.maxlag*efs);

  if strcmp(config.method,'TRF')
    envelope = CreateLoudnessFeature(audio.data,audio.fs,efs);
    stop = min(stop,length(envelope));

    [~,prediction] = FindTRF([],[],-1,eeg_data(:,start:stop)',weights,...
                             lags,'Shrinkage');

    cor_data = [prediction envelope(start:stop,:)];
  elseif strcmp(config.method,'CCA')
    spect = CreateAudiospectFeature(audio.data,audio.fs,efs);
    eeg_lagged = LagGenerator(eeg(:,start:stop)',lags);
    prediction = eeg_lagged * weights.eeg;
    audio_proj = spect * weights.spect;

    cor_data = [prediction audio_proj];
  end
end
