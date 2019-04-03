function fake = trf_fake_data(eeg,stim_info,eps,first_channel,last_channel,filter_fn,stim_fn)
    N = sum(arrayfun(filter_fn,1:length(eeg.trial)));
    j = 0;
    fake_eeg = cell(size(eeg.trial));

    textprogressbar('generating fake data...');
    onCleanup(@() textprogressbar(''));
    for i = 1:length(eeg.trial)
        if filter_fn(i)
            stim = stim_fn(i);
            stim_envelope = CreateLoudnessFeature(stim,stim_info.fs,eeg.fsample);
            response = eeg.trial{i};

            min_len = min(size(stim_envelope,1),size(response,2));
            stim_envelope = stim_envelope(1:min_len);
            cur_fake = normrnd(0,eps*mean(stim_envelope),size(response));

            % keyboard
            cur_fake(first_channel:last_channel,1:min_len) = ...
            bsxfun(@plus,cur_fake(first_channel:last_channel,1:min_len),stim_envelope');

            fake_eeg{i} = cur_fake;

            j = j+1;
            textprogressbar(100*(j/N));
        end
    end

    fake = [];
    fake.fsample = eeg.fsample;1
    fake.trial = fake_eeg;
end
