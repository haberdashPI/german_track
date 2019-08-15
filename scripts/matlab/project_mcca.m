function eeg = project_mcca(eeg,nkeep,iA,mu)
    selection = zeros(size(iA,2),1);
    selection(1:nkeep) = 1;
    for t = 1:length(eeg.trial)
        arr = eeg.raw_trial{t};
        proj_arr = arr';
        proj_arr = proj_arr - mu;
        proj_arr = proj_arr * (iA*diag(selection)*pinv(iA));
        eeg.trial{t} = (proj_arr + mu)';
    end
end
