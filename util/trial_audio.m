function result = trial_audio(info,stim_event,config)
  global data_dir

  stim_num = stim_event.sound_index;
  target_time = info.test_block_cfg.target_times(stim_num);

  if strcmp(config.range,'none')
    start = 0;
    stop = Inf;
  elseif target_time > 0
    start = target_time + config.range(1);
    stop = target_time + config.range(2);
  else
    result = empty_audio(info,target_time);
    return
  end

  audio_file = fullfile(data_dir,'audio','trials',sprintf('trial_%d.wav',stim_num));
  [stim,fs] = audioread(audio_file);
  sent_idx = info.test_block_cfg.trial_sentences(stim_num,:);

  if ~config.filter(info,stim_event)
    result = empty_audio(info,target_time);
    return
  end

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
  elseif strcmp(config.label,'global') || strcmp(config.label,'mixture') || strcmp(config.label,'test')
    data = stim(:,1) + stim(:,2);
  elseif strcmp(config.label,'feature')
    data = stim(:,2);
  elseif strcmp(config.label,'object')
    new_config = config;
    new_config.label = 'male';
    result = trial_audio(info,stim_event,new_config);
    return
  elseif strcmp(config.label,'condition')
    new_config = config;
    new_config.label = stim_event.condition{:};
    result = trial_audio(info,stim_event,new_config);
    return
  elseif strcmp(config.label,'')
  elseif strcmp(config.label,'non_targets')
    a_config = config;
    b_config = config;
    a_config.label = 'non_target1';
    b_config.label = 'non_target2';

    a = trial_audio(info,stim_event,a_config);
    b = trial_audio(info,stim_event,b_config);
    if length(a.data) <= length(b.data)
      a.data(end+1:length(b.data)) = 0.0;
    else
      b.data(end+1:length(a.data)) = 0.0;
    end
    a.data = a.data + b.data;
    result = a;
    return
  else
    error(['Unexpected label "' config.label '".'])
  end

  if fs ~= info.fs
    error('Sample rates of experiment metadata and wav files do not match.');
  end

  result.target_time = target_time;
  result.start = start;
  result.stop = stop;
  result.fs = fs;
  result.data = data;
end

function result = empty_audio(info,target_time)
  result.target_time = target_time;
  result.start = 0;
  result.stop = 0;
  result.fs = info.fs;
  result.data = [];
end
