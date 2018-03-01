function stim_data = get_stim_data(info,stim_event)
  stim_num = stim_event.sound_index;
  if stim_num < 1 || stim_num > info.test_block_cfg.num_trials
    error(['Stimulus index must fall between 1 and ' ...
           num2str(info.test_block_cfg.num_trials)])
  end

  [stim,fs] = audioread(['audio/trials/trial_' num2str(stim_num) '.wav']);

  % target information
  target_idx = info.test_block_cfg.trial_dev_speakers(stim_num);
  target_labels = {'male','fem_young','fem_old'};
  if target_idx > 0
    stim_data.target = target_labels{target_idx};
  else
    stim_data.target = 'none';
  end
  stim_data.target_time = info.test_block_cfg.target_times(stim_num);

  % get the index of the stimulus sentence
  sent_idx = info.test_block_cfg.trial_sentences(stim_num,:);

  % target hits
  if strcmp(stim_event.response{:},'2') && stim_data.target_time > 0
    stim_data.stream.hit_target.data = ...
        info.all_sentences{1}{sent_idx(target_idx),1};
  else
    stim_data.stream.hit_target.data = [];
  end
  stim_data.stream.hit_target.fs = fs;
  stim_data.stream.hit_target.start = stim_data.target_time - 1;
  stim_data.stream.hit_target.stop = stim_data.target_time + 0.25;

  % target misses
  if strcmp(stim_event.response{:},'3') && stim_data.target_time > 0
    stim_data.stream.miss_target.data = stim(:,1)+stim(:,2);
  else
    stim_data.stream.miss_target.data = [];
  end
  stim_data.stream.miss_target.fs = fs;
  stim_data.stream.miss_target.start = stim_data.target_time - 1;
  stim_data.stream.miss_target.stop = stim_data.target_time + 0.25;

  % global attention
  if strcmp(stim_event.condition,'test')
    stim_data.stream.test_condition.data = stim(:,1)+stim(:,2);
  else
    stim_data.stream.test_condition.data = [];
  end
  stim_data.stream.test_condition.fs = fs;
  stim_data.stream.test_condition.start = 0;
  stim_data.stream.test_condition.stop = Inf;

  % object attention
  if strcmp(stim_event.condition,'object')
    stim_data.stream.object_condition.data = info.all_sentences{1}{sent_idx(1),1};
  else
    stim_data.stream.object_condition.data = [];
  end
  stim_data.stream.object_condition.fs = fs;
  stim_data.stream.object_condition.start = 0;
  stim_data.stream.object_condition.stop = Inf;

  % feature attention
  if strcmp(stim_event.condition,'feature')
    stim_data.stream.feature_condition.data = stim(:,2);
  else
    stim_data.stream.feature_condition.data = [];
  end
  stim_data.stream.feature_condition.fs = fs;
  stim_data.stream.feature_condition.start = 0;
  stim_data.stream.feature_condition.stop = Inf;

  % the entire sound
  stim_data.stream.mixed.data = stim(:,1)+stim(:,2);
  stim_data.stream.mixed.fs = fs;
  stim_data.stream.mixed.start = 0;
  stim_data.stream.mixed.stop = Inf;

  % sounds by spatial location
  stim_data.stream.left.data = stim(:,1);
  stim_data.stream.left.fs = fs;
  stim_data.stream.left.start = 0;
  stim_data.stream.left.stop = Inf;

  stim_data.stream.right.data = stim(:,2);
  stim_data.stream.right.fs = fs;
  stim_data.stream.right.start = 0;
  stim_data.stream.right.stop = Inf;

  % % sounds by speaker
  stim_data.stream.male.data = info.all_sentences{1}{sent_idx(1),1};
  stim_data.stream.male.fs = fs;
  stim_data.stream.male.start = 0;
  stim_data.stream.male.stop = Inf;

  stim_data.stream.fem_young.data = info.all_sentences{2}{sent_idx(2),1};
  stim_data.stream.fem_young.fs = fs;
  stim_data.stream.fem_young.start = 0;
  stim_data.stream.fem_young.stop = Inf;

  stim_data.stream.fem_old.data = info.all_sentences{3}{sent_idx(3),1};
  stim_data.stream.fem_old.fs = fs;
  stim_data.stream.fem_old.start = 0;
  stim_data.stream.fem_old.stop = Inf;
end
