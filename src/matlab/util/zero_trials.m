function eeg = zero_trials(eeg,bad_trials)
    for i = 1:length(eeg.trial)
        if any(i == bad_trials)
            eeg.trial{i}(:,:) = 0;
        end
    end
end
