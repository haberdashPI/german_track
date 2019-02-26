function [stim_result,eeg_result] = prepare_data(stim,eeg,features,lags)
  [start,stop] = find_range(info,stim_event,config,...
                            eeg.trial{trial},eeg.fsample);
  if strcmp(config.method,'TRF')
    if config.fake_data
      error('Fake data generation not supported for this method.')
    end

    [stim,fs] = trial_audio(info,stim_event,config);
    stim_result = CreateLoudnessFeature(stim,fs,eeg.fsample);
    stop = min(stop,length(stim_result));

    eeg_result = eeg.trial{trial}(:,start:stop)';
    stim_result = stim_result(start:stop);
  elseif strcmp(config.method,'CCA_envelope')
    stim_result = zeros();
    lags = 0:round(config.maxlag*eeg.fsample);

    for env_index = 1:length(config.label)
      config_label = config;
      config_label.label = config.label{env_index};
      [stim,fs] = trial_audio(info,stim_event,config_label);
      envelope = CreateLoudnessFeature(stim,fs,eeg.fsample);
      stop = min(stop,length(envelope));

      stim_result(1:(stop-start+1),env_index) = envelope(start:stop);
    end
    stim_result = stim_result(1:(stop-start+1),:);

    if ~config.fake_data
      eeg_result = LagGenerator(eeg.trial{trial}(:,start:stop)',lags);
    else
      rng(config.fake_seed(trial));
      fake = zeros(size(eeg.trial{trial}(:,start:stop)'));
      for i = 1:length(config.fake_channels)
        for j = config.fake_channels{i}
          N = min(max(0,size(fake,1) - config.fake_lag),size(stim_result,1));
          fake(1:(N+config.fake_lag),j) = ...
            [zeros(config.fake_lag,1); stim_result(1:N,i)];
        end
      end
      eeg_result = LagGenerator(fake + randn(size(fake))*config.fake_noise,lags);
    end
  elseif strcmp(config.method,'CCA')
    if config.fake_data
      error('Fake data generation not supported for this method.')
    end
    lags = 0:round(config.maxlag*eeg.fsample);
    [stim,fs] = trial_audio(info,stim_event,config_label);
    stim_result = CreateAudiospectFeature(stim,fs,eeg.fsample);
    stop = min(stop,size(spect,1));
    eeg_result = LagGenerator(eeg.trial{trial}(:,start:stop)',lags);
    stim_result = stim_result(start:stop,:);
  end
end
