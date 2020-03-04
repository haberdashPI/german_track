function eeg = my_rmeog(eeg,eog_channels,bad_trials)
    selection = [];
    selection.channel = eog_channels;
    eog = ft_selectdata(selection,eeg);
    textprogressbar('Removing EOG artifacts...');
    onCleanup(@() textprogressbar('\n'));
    for i = 1:length(eeg.trial)
        data = eeg.trial{i}';
        eogdat = eog.trial{i}';
        if ~any(i == bad_trials)
            eeg.trial{i} = (data - eogdat * (eogdat\data))';
        end
        textprogressbar(100*(i/length(eeg.trial)));
    end
    fprintf('\n')
end
