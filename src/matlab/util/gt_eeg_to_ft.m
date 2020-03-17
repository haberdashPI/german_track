function eeg = gt_eeg_to_ft(trial,label,sr)
    eeg = [];
    eeg.trial = trial;
    eeg.label = label;
    eeg.time = cellfun(@(x) (1:size(x,2)) / sr, trial, 'UniformOutput', false);
    sinfo = zeros(length(eeg.time),2);
    sinfo(:,2) = cumsum(cellfun(@length,eeg.time));
    sinfo(2:end,1) = sinfo(1:(end-1),2)+1;
    sinfo(1,1) = 1;
    eeg.sampleinfo = sinfo;
end
