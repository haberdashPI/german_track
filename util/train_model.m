function model = train_model(eeg,info,stim_events,train_config,trials)
% model = train_model(eeg,info,stim_events,train_config,trials)
%
% Train model using a given method
%
% Parameters:
%
% eeg - a structre with two fields, consistent with fieldtrip's return values
%   fsample - the sample rate of the eeg data
%   trial - a cell array of trials. Each trial is an TxN matrix where
%      T is the number of time slices and N the number of channels
% info - a large structure of predefined experiment configuartion
%    information loaded from the file 'exeriment_record.mat'
% stim_events - a table detailing experiment stimulus events
%    sample - the sample where the stimulus began
%    time - the time where the stimulus began
%    condition - the experimental condition of this stimulus trial, either
%        'test' (global attending), 'object' (object-based attention) or
%        'feature' (feature-based attention).
%    response - '2' (heard a target) or '3' (did not hear a target)
%    sound_index - an index 1-50 indicating the specific stimulus presented
% train_config - a structure defining the method of training with the following
% fields:
%   train - an array of strcutures with the following fields
%       model_name - a description of this model
%       maxlag - the maximum lag, in seconds, to consider of the eeg data.
%       method - 'TRF' or 'CCA' indicating how to decode eeg data
%       method_params - a method dependent structure
%            - for 'TRF' - no fields
%            - for 'CCA'
%                 - regular - the degree of regularization
%       range - the range (in seconds) around the target event (if it exists) to
%          use. Otherwise, all data is used.
%       filter - a function, used to select which trials to train on.
%          is passed the 'info' structure' and the row of the
%          stim_events data structure of relevance.
%   test - an array of structures with the following fields
%       models - a list of trained models (from above) to test on
%       label - the label used to select the audio data (see 'trial_audio')
%
%       + the fields 'maxlag', 'method', 'method_params', 'range',
%         and 'filter' from above
  for i = 1:length(train_config)
    [stim,response] = select_data(eeg,info,stim_events,train_config(i),trials);

    weights = train_helper(train_config(i),stim,response,eeg.fsample);
    model.weights.(train_config(i).model_name) = weights;
  end
end

function [all_audio,all_eeg] = select_data(eeg,info,stim_events,config,trials)
  if isempty(trials)
    error('No training trials provided')
  end

  % put this in its own function?
  if strcmp(config.method,'TRF')
    all_audio = zeros(size(eeg.trial{1},2)*length(trials),1);
    all_eeg = zeros(size(eeg.trial{1},2)*length(trials),size(eeg.trial{1},1));
  elseif strcmp(config.method,'CCA_envelope')
    lags = 0:round(config.maxlag*eeg.fsample);
    all_audio = zeros(size(eeg.trial{1},2)*length(trials),length(config.label));
    all_eeg = zeros(size(eeg.trial{1},2)*length(trials),...
                    length(lags)*size(eeg.trial{1},1));
  elseif strcmp(config.method,'CCA')
    lags = 0:round(config.maxlag*eeg.fsample);
    all_audio = zeros(size(eeg.trial{1},2)*length(trials),128);
    all_eeg = zeros(size(eeg.trial{1},2)*length(trials),...
                    length(lags)*size(eeg.trial{1},1));
  end

  t = 1;
  for trial = trials
    if config.filter(info,stim_events(trial,:))
      [stim_result,eeg_result] = prepare_data(info,config,...
        stim_events(trial,:),eeg,trial);

      all_eeg(t:(t+size(eeg_result,1)-1),1:size(eeg_result,2)) = eeg_result;
      all_audio(t:(t+size(stim_result,1)-1),1:size(stim_result,2)) = stim_result;
      t = t+size(eeg_result,1);
    end
  end

  if t == 1
    error('Training filter removed all trials!')
  end

  all_audio = all_audio(1:t,:);
  all_eeg = all_eeg(1:t,:);
end

function model = train_helper(config,stim,response,efs)
  if strcmp(config.method,'TRF')
    lags = 0:round(config.maxlag*efs);
    model = FindTRF(stim,response,-1,[],[],lags,'Shrinkage');
  elseif strcmp(config.method,'CCA') || strcmp(config.method,'CCA_envelope')
    [Wspect,Weeg] = cca(stim',response',config.method_params.regular);
    model = [];
    model.spect = Wspect;
    model.eeg = Weeg;
  end
end
