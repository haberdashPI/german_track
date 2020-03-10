function eeg = gt_downsample(eeg,stim_events,n)
    eeg = gt_settrials(@nt_dsample,eeg,n,'progress','resampling...');
    eeg.hdr.Fs = eeg.hdr.Fs / n;
    eeg.time = gt_fortrials(@(data) ((1:size(data,1))/eeg.hdr.Fs),eeg)';
    eeg.fsample = eeg.fsample / n;
    eeg.sampleinfo = round([stim_events.time*eeg.fsample ...
        stim_events.time*eeg.fsample + ...
            (cellfun(@(x) size(x,2),eeg.trial))' - 1]);
end
