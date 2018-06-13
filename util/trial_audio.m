function [data,fs] = trial_audio(info,stim_event,config)
  global data_dir

  if iscell(config.label)
    error(['Multiple labels provided, did you mean to call trial_audio'...
           ' on them seperately?'])
  end
  stim_num = stim_event.sound_index;

  audio_file = fullfile(data_dir,'audio','trials',sprintf('trial_%d.wav',stim_num));
  [stim,fs] = audioread(audio_file);
  sent_idx = info.test_block_cfg.trial_sentences(stim_num,:);

  if strcmp(config.label,'male')
    data = info.all_sentences{1}{sent_idx(1),1};
  elseif strcmp(config.label,'fem_young')
    data = info.all_sentences{2}{sent_idx(2),1};
  elseif strcmp(config.label,'fem_old')
    data = info.all_sentences{3}{sent_idx(3),1};
  elseif strcmp(config.label,'target')
    target_idx = info.test_block_cfg.trial_dev_speakers(stim_num);
    data = info.all_sentences{target_idx}{sent_idx(target_idx),1};
  elseif strcmp(config.label,'non_target1')
    target_idx = info.test_block_cfg.trial_dev_speakers(stim_num);
    nts = 1:3;
    nts(nts ~=  sent_idx(target_idx));
    data = info.all_sentences{nts(1)}{sent_idx(nts(1)),1};
  elseif strcmp(config.label,'non_target2')
    target_idx = info.test_block_cfg.trial_dev_speakers(stim_num);
    nts = 1:3;
    nts(nts ~=  sent_idx(target_idx));
    data = info.all_sentences{nts(2)}{sent_idx(nts(2)),1};
  elseif strcmp(config.label,'global') ...
    || strcmp(config.label,'mixture') ...
    || strcmp(config.label,'test')

    data = stim(:,1) + stim(:,2);
  elseif strcmp(config.label,'feature')
    data = stim(:,2);
  elseif strcmp(config.label,'object')
    new_config = config;
    new_config.label = 'male';
    [data,fs] = trial_audio(info,stim_event,new_config);
  elseif strcmp(config.label,'condition')
    new_config = config;
    new_config.label = stim_event.condition{:};
    [data,fs] = trial_audio(info,stim_event,new_config);
  elseif strcmp(config.label,'non_targets')
    a_config = config;
    b_config = config;
    a_config.label = 'non_target1';
    b_config.label = 'non_target2';

    [a_data,a_fs] = trial_audio(info,stim_event,a_config);
    [b_data,b_fs] = trial_audio(info,stim_event,b_config);
    if length(a.data) <= length(b.data)
      a_data(end+1:length(b_data)) = 0.0;
    else
      b_data(end+1:length(a_data)) = 0.0;
    end
    data = a_data+b_data;
    if a_fs ~= b_fs
      error('Unexpected mismatched sample rates')
    end
    fs = a_fs;
  else
    error(['Unexpected label "' config.label '".'])
  end

  if fs ~= info.fs
    error('Sample rates of experiment metadata and wav files do not match.');
  end
end
