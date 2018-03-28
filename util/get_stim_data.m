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
  hit_target.test_needs_target = 1;
  if stim_data.target_time > 0
    hit_target.test_data = ...
        info.all_sentences{target_idx}{sent_idx(target_idx),1};
  else
    hit_target.test_data = [];
  end
  if strcmp(stim_event.response{:},'2') && stim_data.target_time > 0
    hit_target.data = hit_target.test_data;
  else
    hit_target.data = [];
  end
  hit_target.fs = fs;
  hit_target.start = stim_data.target_time - 2;
  hit_target.stop = stim_data.target_time + 0;
  hit_target.test_start = stim_data.target_time - 2;
  hit_target.test_stop = stim_data.target_time + 0;
  stim_data.stream.hit_target = hit_target;

  % target misses
  miss_mix.test_data = stim(:,1)+stim(:,2);
  miss_mix.test_needs_target = 1;
  if strcmp(stim_event.response{:},'3') && stim_data.target_time > 0
    miss_mix.data = miss_mix.test_data;
  else
    miss_mix.data = [];
  end
  miss_mix.fs = fs;
  miss_mix.start = stim_data.target_time - 2;
  miss_mix.stop = stim_data.target_time + 0;
  miss_mix.test_start = stim_data.target_time - 2;
  miss_mix.test_stop = stim_data.target_time + 0;
  stim_data.stream.miss_mix = miss_mix;

  stim_data.stream.miss_nontarget.test_needs_target = 1;
  if target_idx > 0
    nts = 1:3;
    nts(nts ~=  sent_idx(target_idx));
    non_target1 = info.all_sentences{nts(1)}{sent_idx(nts(1)),1};
    non_target2 = info.all_sentences{nts(2)}{sent_idx(nts(2)),1};
    % padding
    if length(non_target1) > length(non_target2)
      non_target2(end+1:length(non_target1)) = 0;
    elseif length(non_target2) > length(non_target1)
      non_target1(end+1:length(non_target2)) = 0;
    end
    miss_nontarget.test_data = non_target1 + non_target2;
  else
     miss_nontarget.test_data = [];
  end
  if strcmp(stim_event.response{:},'3') && stim_data.target_time > 0
    miss_nontarget.data = ...
        miss_nontarget.test_data;
  else
    miss_nontarget.data = [];
  end
  miss_nontarget.fs = fs;
  miss_nontarget.test_start = stim_data.target_time - 2;
  miss_nontarget.test_stop = stim_data.target_time + 0;
  miss_nontarget.start = stim_data.target_time - 2;
  miss_nontarget.stop = stim_data.target_time + 0;
  stim_data.stream.miss_nontarget = miss_nontarget;

  % global attention
  stim_data.stream.test_condition.test_needs_target = 0;
  test_condition.test_data = stim(:,1)+stim(:,2);
  if strcmp(stim_event.condition,'test')
    test_condition.data = ...
        test_condition.test_data;
  else
    test_condition.data = [];
  end
  test_condition.fs = fs;
  test_condition.start = 0;
  test_condition.stop = Inf;
  test_condition.test_start = 0;
  test_condition.test_stop = Inf;
  stim_data.stream.test_condition = test_condition;

  % object attention
  stim_data.stream.object_condition.test_needs_target = 0;
  object_condition.test_data = ...
      info.all_sentences{1}{sent_idx(1),1};
  if strcmp(stim_event.condition,'object')
    object_condition.data = ...
        object_condition.test_data;
  else
    object_condition.data = [];
  end
  object_condition.fs = fs;
  object_condition.start = 0;
  object_condition.stop = Inf;
  object_condition.test_start = 0;
  object_condition.test_stop = Inf;
  stim_data.stream.object_condition = object_condition;

  % feature attention
  stim_data.stream.feature_condition.test_needs_target = 0;
  feature_condition.test_data = stim(:,2);
  if strcmp(stim_event.condition,'feature')
    feature_condition.data = ...
        feature_condition.test_data;
  else
    feature_condition.data = [];
  end
  feature_condition.fs = fs;
  feature_condition.start = 0;
  feature_condition.stop = Inf;
  feature_condition.test_start = 0;
  feature_condition.test_stop = Inf;
  stim_data.stream.feature_condition = feature_condition;

  % train a model on all trials, assuming
  % 1. we can find some general means to extract the envelope
  % 2. the listener follows the directions exactly
  general_envelope.test_needs_target = 1;
  if strcmp(stim_event.condition,'feature')
    general_envelope.data = stim(:,2);
  elseif strcmp(stim_event.condition,'test')
    general_envelope.data = stim(:,2)+stim(:,1);
  elseif strcmp(stim_event.condition,'object')
    general_envelope.data = info.all_sentences{1}{sent_idx(1),1};
  end
  if stim_data.target_time > 0
    general_envelope.test_data = ...
        info.all_sentences{target_idx}{sent_idx(target_idx),1};
  end
  general_envelope.fs = fs;
  general_envelope.test_start = stim_data.target_time - 2;
  general_envelope.test_stop = stim_data.target_time + 0;
  general_envelope.start = 0;
  general_envelope.stop = Inf;
  stim_data.stream.general_envelope = general_envelope;

  % % the entire sound
  % stim_data.stream.mixed.data = stim(:,1)+stim(:,2);
  % stim_data.stream.mixed.fs = fs;
  % stim_data.stream.mixed.start = 0;
  % stim_data.stream.mixed.stop = Inf;
  % stim_data.stream.mixed.test_start = 0;
  % stim_data.stream.mixed.test_stop = Inf;

  % % sounds by spatial location
  % stim_data.stream.left.data = stim(:,1);
  % stim_data.stream.left.fs = fs;
  % stim_data.stream.left.start = 0;
  % stim_data.stream.left.stop = Inf;
  % stim_data.stream.left.test_start = 0;
  % stim_data.stream.left.test_stop = Inf;

  % stim_data.stream.right.data = stim(:,2);
  % stim_data.stream.right.fs = fs;
  % stim_data.stream.right.start = 0;
  % stim_data.stream.right.stop = Inf;
  % stim_data.stream.right.test_start = 0;
  % stim_data.stream.right.test_stop = Inf;

  % % % sounds by speaker
  % stim_data.stream.male.data = info.all_sentences{1}{sent_idx(1),1};
  % stim_data.stream.male.fs = fs;
  % stim_data.stream.male.start = 0;
  % stim_data.stream.male.stop = Inf;
  % stim_data.stream.male.test_start = 0;
  % stim_data.stream.male.test_stop = Inf;

  % stim_data.stream.fem_young.data = info.all_sentences{2}{sent_idx(2),1};
  % stim_data.stream.fem_young.fs = fs;
  % stim_data.stream.fem_young.start = 0;
  % stim_data.stream.fem_young.stop = Inf;
  % stim_data.stream.fem_young.test_start = 0;
  % stim_data.stream.fem_young.test_stop = Inf;

  % stim_data.stream.fem_old.data = info.all_sentences{3}{sent_idx(3),1};
  % stim_data.stream.fem_old.fs = fs;
  % stim_data.stream.fem_old.start = 0;
  % stim_data.stream.fem_old.stop = Inf;
  % stim_data.stream.fem_old.test_start = 0;
  % stim_data.stream.fem_old.test_stop = Inf;
end
