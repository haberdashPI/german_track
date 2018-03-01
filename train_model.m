function model = train_model(eeg,stim_events,train_config,streams,fileprefix,usecache)
  if nargin < 4
    usecache = 0;
  end

  model = {};

  textprogressbar('processing trials: ');
  for trial = 1:length(eeg.trial)
    modelfile = [fileprefix '_trial_' num2str(trial,'%03d') '.mat'];

    if usecache && exist(modelfile)
      textprogressbar(100*(trial / length(eeg.trial)));
      mf = load(modelfile);
      model{trial} = mf.trial_model;
    else
      trial_config = train_config.trial{trial};

      trial_model = [];
      % streams = fieldnames(trial_config.stream);
      N = length(eeg.trial)*length(streams);
      lags = 0:round(0.25*eeg.fsample);
      for i = 1:length(streams)
        textprogressbar(100*((length(streams)*(trial-1) + i) / N));
        stream_name = streams{i};
        trf = ...
            train_helper(trial_config.stream.(stream_name),...
                         eeg.trial{trial},eeg.fsample,lags);
        trial_model.trf.(stream_name) = trf;
      end
      trial_model.lags = lags;
      trial_model.target = trial_config.target;
      trial_model.target_time = trial_config.target_time;
      save(modelfile,'trial_model');
      model{trial} = trial_model;
    end
  end
  textprogressbar('finished!');
end

function model = train_helper(stim,eeg,efs,lags)
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
