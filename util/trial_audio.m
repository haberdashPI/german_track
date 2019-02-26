function [data,fs] = trial_audio(config,trial,source)
% trial_audio    Find the time-amplitude data for a given trial and source
%
% [data,fs] = trial_audio(config,trial,source)
% config - The experiment configuration, created using config_experiment
% trial - An index which indicates the trial from which we're requesting audio
% source - The name of the source: 'male', 'fem_young' or 'fem_old'.

  stim_num = config.stim_events(trial,:).sound_index;
  sent_idx = config.info.test_block_cfg.trial_sentences(stim_num,:);

  if strcmp(source,'male')
    data = config.info.all_sentences{1}{sent_idx(1),1};
  elseif strcmp(source,'fem_young')
    data = config.info.all_sentences{2}{sent_idx(2),1};
  elseif strcmp(source,'fem_old')
    data = config.info.all_sentences{3}{sent_idx(3),1};
  else
    error(['Unexpected source "' source '".'])
  end
end
