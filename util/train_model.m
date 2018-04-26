function model = train_model(eeg,info,stim_events,train_config,fileprefix,usecache)
  if nargin < 6
    usecache = 0;
  end

  model = {};

  % try
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
        for i = 1:length(train_config)
          stim = trial_audio(info,stim_events(trial,:),train_config(i));
          textprogressbar(100*((length(train_config)*(trial-1) + i) / N));
          weights = train_helper(train_config(i),stim,eeg.trial{trial},eeg.fsample);
          trial_model.weights.(train_config(i).model_name) = weights;
        end
        save(modelfile,'trial_model');
        model{trial} = trial_model;
      end
    end
    textprogressbar('finished!');
  % catch e
  %   textprogressbar('error!');
  %   rethrow(e);
  % end
end

function model = train_helper(config,stim,eeg,efs)
  start = max(1,floor(efs * stim.start));
  stop = min(ceil(efs * stim.stop),size(eeg,2));

  lags = 0:round(config.maxlag*efs);
  if isempty(stim.data)
    model = [];
  elseif strcmp(config.method,'TRF')
    envelope = CreateLoudnessFeature(stim.data,stim.fs,efs);
    stop = min(stop,length(envelope));

    model = FindTRF(envelope(start:stop),eeg(:,start:stop)',-1,[],[],lags, ...
                    'Shrinkage');
  elseif strcmp(config.method,'CCA')
    spect = CreateAudiospectFeature(stim.data,stim.fs,efs);
    stop = min(stop,size(spect,1));
    
    eeg_lagged = LagGenerator(eeg(:,start:stop)',lags);
    
    keyboard;

    [Wspect,Weeg] = cca(spect(start:stop,:)',eeg_lagged',config.method_params.regular);
    model = [];
    model.spect = Wspect;
    model.eeg = Weeg;
  end
end
