function eeg = my_channelrepair(cfg,eeg)
    trials = 1:length(eeg.trial);
    if isfield(cfg,'trials') && ~strcmp(cfg.trials,'all')
        trials = cfg.trials;
    end
    new_eeg = ft_channelrepair(cfg,eeg);

    j = 1;
    for i = trials
        eeg.trial{i}(1:64,:) = new_eeg.trial{j}(1:64,:);
        j = j+1;
    end
end
