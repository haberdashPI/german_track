function audio = sentence(stim_events,stim_info,i,j)
  stim_num = stim_events{i,'sound_index'};
  sent_idx = stim_info.test_block_cfg.trial_sentences(stim_num,:);

  % TODO: load the audio file here (not stored in config structure anymore)
  audio = stim_info.all_sentences{j}{sent_idx(j),1};
end
