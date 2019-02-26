function [envelopes,eeg_result] = prepare_envelopes(config,stims,eeg)
% prepare_envelopes   Prepare the audio and eeg for an envelope based analysis
%
% [envelopes,eeg] = prepare_envelopes(config,stims,eeg)
% config - The experiment configuration, created using config_experiment
% stims - cell array of audio stimuli whose envelopes are to be extracted
% eeg - a cell array of trials. Each trial is an TxN matrix where
%      T is the number of time slices and N the number of channels
%
% returns:
% envelopes - the envelopes as a TxM matrix, where T is the number of time slices
%    and M is length(stims)
% eeg - the eeg data of size TxN. Note that T may differ from the input
%   so that both evenlopes and eeg have the same time span (any
%   samples with no matching audio are silently dropped). N
%   may be equal to the nubmer of channels, but it may also
%   include augmented features (e.g. lagged values), depending on `config`.
%   Depending on `config`, the returned eeg may also be simulated data.

  % the minimum length signal
  T = size(eeg,1);
  for i = 1:length(stims)
    T = min(T,size(stims{i},1));
  end

  envelopes = zeros(T,length(stims));
  for i = 1:length(stims)
    envelopes(:,i) = CreateLoudnessFeature(stims{i}(1:T),fs,eeg.fsample);
  end

  if config.fake_data
    rng(config.fake_seed(trial));
    fake = zeros(T,size(eeg,2));
    for i = 1:length(config.fake_channels)
      for j = config.fake_channels{i}
        N = max(0,size(fake,1) - config.fake_lag);
        fake(1:(N+config.fake_lag),j) = ...
          [zeros(config.fake_lag,1); stim_result(1:N,i)];
      end
    end
    eeg_result = fake + randn(size(fake))*config.fake_noise;
  else
    eeg_result = eeg(1:T,:);
  end

  if config.explicit_lags
    eeg_result = LagGenerator(eeg_result,config.lags);
  end
end
