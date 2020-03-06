
function eeg = gt_interpolate_bad_channels(eeg,bad_indices,coords)
    [toGood,fromGood] = nt_interpolate_bad_channels(eeg,bad_indices,coords);
    eeg = eeg*(toGood*fromGood);
end
