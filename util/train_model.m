function model = train_model(eeg,info,stim_events,train_config,fileprefix,usecache)
  if nargin < 6
    usecache = 0;
  end

  model = {};

  try
    textprogressbar('processing trials: ');
    for trial = 1:length(eeg.trial)
      modelfile = sprintf('%s_trial%03d.mat',fileprefix,trial);

      if usecache && exist(modelfile)
        textprogressbar(100*(trial / length(eeg.trial)));
        mf = load(modelfile);
        model{trial} = mf.trial_model;
      else
        trial_model = [];
        % streams = fieldnames(trial_config.stream);
        N = length(eeg.trial)*length(train_config);
        lags = 0:round(0.25*eeg.fsample);
        for i = 1:length(train_config)
          stim = trial_audio(info,stim_events(trial,:),train_config(i));
          textprogressbar(100*((length(train_config)*(trial-1) + i) / N));
          trf = train_helper(train_config(i),stim,eeg.trial{trial},eeg.fsample,lags);
          trial_model.trf.(train_config(i).model_name) = trf;
        end
        trial_model.lags = lags;
        save(modelfile,'trial_model');
        model{trial} = trial_model;
      end
    end
    textprogressbar('finished!');
  catch e
    textprogressbar('error!');
    rethrow(e);
  end
end

function model = train_helper(config,stim,eeg,efs,lags)
  if isempty(stim.data)
    model = [];
  else
    envelope = CreateLoudnessFeature(stim.data,stim.fs,efs);
    start = max(1,floor(efs * stim.start));
    stop = min(ceil(efs * stim.stop),size(eeg,2));
    if length(envelope) < stop
      envelope(end+1:stop) = 0;
    end

    model = FindTRF(envelope(start:stop),eeg(:,start:stop)',-1,[],[],lags, ...
                    'Shrinkage');
  end
end
