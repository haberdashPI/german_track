function time = get_target_time(info,stim_event)
  stim_num = stim_event.sound_index;
  time = info.test_block_cfg.target_times(stim_num);
end
