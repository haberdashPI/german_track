function eeg = project_mcca(eeg,weights,nkeep,chans,iA,mu)
    selection = 1:nkeep;

    eeg.projected = {};
    iAinv = pinv(iA);
    iAinv = iAinv(selection,:);
    iA = iA(:,selection);
    T = (iA*iAinv);
    eeg.components = iAinv;

    for t = 1:length(eeg.trial)
        arr = eeg.trial{t}(chans,:) .* weights{t}(chans,:);
        eeg.projected{t} = (arr' * iA)';
        eeg.trial{t} = (arr' * T)';
    end
    eeg.label = eeg.label(chans);
end
