function stim_data = get_stim_data(stim_num,info)
  if stim_num < 1 || stim_num > info.test_block_cfg.num_trials
    error(['Stimulus index must fall between 1 and ' ...
           num2str(info.test_block_cfg.num_trials)])
  end

  [stim,fs] = audioread(['audio/trials/trial_' num2str(stim_num) '.wav']);
  stim_data.fs = fs;
  % the entire sound
  stim_data.stream.mixed = stim(:,1)+stim(:,2);

  % sounds by spatial location
  stim_data.stream.left = stim(:,1);
  stim_data.stream.right = stim(:,2);

  % sounds by speaker
  sent_idx = info.test_block_cfg.trial_sentences(stim_num,:);
  stim_data.stream.male = info.all_sentences{1}{sent_idx(1),1};
  stim_data.stream.fem_young = info.all_sentences{2}{sent_idx(2),1};
  stim_data.stream.fem_old = info.all_sentences{3}{sent_idx(3),1};

  % sounds by target
  idx = info.test_block_cfg.trial_dev_speakers(stim_num);
  target_labels = {'male','fem_young','fem_old'};
  if idx > 0
    stim_data.target = target_labels{idx};
  else
    stim_data.target = 'none';
  end

  % NOTE: record info.test_block_cfg.target_times
  stim_data.target_time = info.test_block_cfg.target_times(stim_num);
end
