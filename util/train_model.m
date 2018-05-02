function model = train_model(eeg,info,stim_events,train_config,trials)
  for i = 1:length(train_config)
    [stim,response] = select_data(eeg,info,stim_events,train_config(i),trials);

    weights = train_helper(train_config(i),stim,response,eeg.fsample);
    model.weights.(train_config(i).model_name) = weights;
  end
end

function [all_audio,all_eeg] = select_data(eeg,info,stim_events,config,trials)
  if strcmp(config.method,'TRF')
    all_audio = zeros(size(eeg.trial{1},2)*length(trials),1);
    all_eeg = zeros(size(eeg.trial{1},2)*length(trials),size(eeg.trial{1},1));
  elseif strcmp(config.method,'CCA')
    lags = 0:round(config.maxlag*eeg.fsample);
    all_audio = zeros(size(eeg.trial{1},2)*length(trials),128);
    all_eeg = zeros(size(eeg.trial{1},2)*length(trials),...
                    length(lags)*size(eeg.trial{1},1));
  end

  t = 1;
  for trial = trials
    stim = trial_audio(info,stim_events(trial,:),config);

    if ~isempty(stim.data)
      start = max(1,floor(eeg.fsample * stim.start));
      stop = min(ceil(eeg.fsample * stim.stop),size(eeg.trial{trial},2));

      if strcmp(config.method,'TRF')
        envelope = CreateLoudnessFeature(stim.data,stim.fs,eeg.fsample);
        stop = min(stop,length(envelope));

        cur_eeg = eeg.trial{trial}(:,start:stop)';
        all_eeg(t:t+size(cur_eeg,1),1:size(cur_eeg,2)) = cur_eeg;
        all_audio(t:t+size(all_audio,1),1) = envelope(start:stop);
        t = t+size(cur_eeg,1);
      elseif strcmp(config.method,'CCA')
        spect = CreateAudiospectFeature(stim.data,stim.fs,eeg.fsample);
        stop = min(stop,size(spect,1));

        eeg_lagged = LagGenerator(eeg.trial{trial}(:,start:stop)',lags);

        all_eeg(t:(t+size(eeg_lagged,1)-1),1:size(eeg_lagged,2)) = eeg_lagged;
        all_audio(t:(t+size(eeg_lagged,1)-1),1:size(spect,2)) = spect(start:stop,:);

        t = t+size(eeg_lagged,1);
      end
    end
  end

  all_audio = all_audio(1:t,:);
  all_eeg = all_eeg(1:t,:);
end

function model = train_helper(config,stim,response,efs)
  if strcmp(config.method,'TRF')
    lags = 0:round(config.maxlag*efs);
    model = FindTRF(stim,response,-1,[],[],lags,'Shrinkage');
  elseif strcmp(config.method,'CCA')
    [Wspect,Weeg] = cca(stim',response',config.method_params.regular);
    model = [];
    model.spect = Wspect;
    model.eeg = Weeg;
  end
end
