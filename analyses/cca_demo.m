% x is,
% all observations (time x trial)
% by all of the features (channel x subject)

% TODO: before-hand; remove bad channels, and interpolate
x = x - repmat(mean(x,1), size(x,1),1); % subtract mean from each column
C = x'*x; % covariance matrix

% assuming: same number of channels
[A,score,AA] = nt_mcca(C,nchan);
% score: ~variance acounted for components (slow slope might mean more cleaning needs to happen)

comp = X * A; % MCCA components, where first column is most repeatable component across subjects
nkeep = 20; % number of components to keep

% Project out all but first "nkeep" components
for i = 1:nsub
    arr = <subject data in "tall" form, dim= ntime*ntrial x nchan>
    iA = AA{i}; % subject-specific MCCA weights
    eye_select = zeros(size(iA,2),1);
    eye_select(1:nkeep) = 1;
    y = arr* (iA*diag(eye_select)*pinv(iA));
    % y: MCCA-cleaned data for subject i
end
