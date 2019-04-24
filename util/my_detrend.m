function eeg = my_detrend(eeg,bad_trials)
    % linear detrending
    textprogressbar('Linear Detrending...');
    onCleanup(@() textprogressbar(''));
    for i = 1:length(eeg.trial)
        data = eeg.trial{i}';
        if ~any(i == bad_trials)
            eeg.trial{i} = nt_detrend(data,1)';
        end
        textprogressbar(100*(i/length(eeg.trial)));
    end
    fprintf('\n');

    % polynomial detrending
    textprogressbar('Polynomial Detrending...');
    onCleanup(@() textprogressbar('\n'));
    for i = 1:length(eeg.trial)
        data = eeg.trial{i}';
        if ~any(i == bad_trials)
            eeg.trial{i} = nt_detrend(data,5)';
        end
        textprogressbar(100*(i/length(eeg.trial)));
    end
    fprintf('\n');
end
