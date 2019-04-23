function ordering = sort_trial_times(eeg,stim_events)
    [~,~,cond_i] = unique(stim_events.condition);
    [~,ordering] = sort(stim_events.sound_index + 50*cond_i);
end
