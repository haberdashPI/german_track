function model = train_model(eeg,stim_events,fileprefix,usecache)
  if nargin < 4
    usecache = 0;
  end

  dat = load('config/experiment_record.mat');
  all_stim_data = dat.experiment_cfg;
  model = {};

  textprogressbar('processing trials: ');
  for trial = 1:length(eeg.trial)
    modelfile = [fileprefix '_trial_' num2str(trial,'%03d') '.mat'];

    if usecache && exist(modelfile)
      textprogressbar(100*(trial / length(eeg.trial)));
      mf = load(modelfile);
      stim_data = get_stim_data(stim_events{trial,'sound_index'},all_stim_data);

      %%%%
      % temporary repair code (remove after finishing 0001)
      trial_model = [];
      trial_model.lags = mf.trial_model.lags;
      trial_model.target = mf.trial_model.target;
      trial_model.target_time = mf.trial_model.target_time;
      names = fieldnames(stim_data.stream);
      for i = 1:length(names)
        stream_name = names{i};
        trial_model.trf.(stream_name) = mf.trial_model.(stream_name);

        envelope = envelope_helper(stim_data.stream.(stream_name),all_stim_data.fs,...
                        eeg.trial{trial},eeg.fsample);

        trial_model.envelope.(stream_name) = envelope;
      end
      %%%%

      model{trial} = trial_model;
    else
      stim_data = get_stim_data(stim_events{trial,'sound_index'},all_stim_data);

      trial_model = [];
      names = fieldnames(stim_data.stream);
      N = length(eeg.trial)*length(names);
      lags = 0:round(0.25*eeg.fsample);
      for i = 1:length(names)
        textprogressbar(100*((length(names)*(trial-1) + i) / N));
        stream_name = names{i};
        [model,envelope] = ...
            train_helper(stim_data.stream.(stream_name),all_stim_data.fs,...
                         eeg.trial{trial},eeg.fsample,lags);
        trial_model.trf.(stream_name) = model
        trial_model.envelope.(stream_name) = envelope
      end
      trial_model.lags = lags;
      trial_model.target = stim_data.target;
      trial_model.target_time = stim_data.target_time;
      save(modelfile,'trial_model');
      model{trial} = trial_model;
    end
  end
  textprogressbar('finished!');
end

function envelope_p = envelope_helper(stim,sfs,eeg,efs)
  envelope = CreateLoudnessFeature(stim,sfs,efs);
  [envelope_p,eeg_p] = match_lengths(envelope,eeg);
end

function [model,envelope_p] = train_helper(stim,sfs,eeg,efs,lags)
  envelope = CreateLoudnessFeature(stim,sfs,efs);
  [envelope_p,eeg_p] = match_lengths(envelope,eeg);

  model = FindTRF(envelope_p,eeg_p',-1,[],[],lags,'Shrinkage');
end

function [a,b] = match_lengths(a,b)
  len = max(length(a),size(b,2));
  if length(a) < len
    a(end+1:len) = 0;
  end
  if size(b,2) < len
    b(:,end+1:len) = 0;
  end
end
