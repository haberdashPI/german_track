function cor_data = test_model(eeg_data,efs,audio,config,weights)
  start,stop = find_range(eeg_data,efs,config.range)

  if strcmp(config.method,'TRF')
    envelope = CreateLoudnessFeature(audio.data,audio.fs,efs);
    stop = min(stop,length(envelope));

    [~,prediction] = FindTRF([],[],-1,eeg_data(:,start:stop)',weights,...
                             lags,'Shrinkage');

    cor_data = [prediction envelope(start:stop,:)];
  elseif strcmp(config.method,'CCA')
    spect = CreateAudiospectFeature(audio.data,audio.fs,efs);
    eeg_lagged = LagGenerator(eeg_data(:,start:stop)',lags);

    prediction = eeg_lagged * weights.eeg;
    audio_proj = spect(start:stop,:) * weights.spect;

    cor_data = [prediction audio_proj];
  end
end
