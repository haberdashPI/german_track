function eeg = project_mcca(eeg,nkeep,chans,iA,mu)
    selection = 1:nkeep;

    eeg.projected = {};
    iAinv = pinv(iA);
    iAinv = iAinv(selection,:);
    iA = iA(:,selection);
    T = (iA*iAinv);
    eeg.components = iAinv;

    for t = 1:length(eeg.trial)
        arr = eeg.trial{t}(chans,:);
        proj_arr = arr';
        proj_arr = proj_arr - mu;
        proj_arr = proj_arr * T;
        eeg.projected{t} = proj_arr * iA;
        eeg.trial{t} = (proj_arr + mu)';
    end
    eeg.label = eeg.label(chans);
end
