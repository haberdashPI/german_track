function [start,stop] = find_range(info,stim_event,config,eeg_data,fs)
  if strcmp(config.range,'none')
    start = 1;
    stop = size(eeg_data,2);
  else
    target_time = get_target_time(info,stim_event);
    if target_time > 0
      start = max(1,floor(fs * (target_time + config.range(1))));
      stop = min(ceil(fs * (target_time + config.range(2))),size(eeg_data,2));
    else
      error('No available target')
    end
  end
end
