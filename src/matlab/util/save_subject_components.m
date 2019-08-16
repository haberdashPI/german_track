function save_subject_components(subj,filename)
    fid = fopen(filename,'w','n','UTF-8');
    try
        % onCleanup(@() fclose(fid));

        % number of channels
        nchan = size(subj.trial{1},1);
        fwrite(fid,nchan,'int32');
        % fprintf('number of channels %d\n',nchan);
        % channel names
        for i = 1:size(subj.trial{1},1)
            nchar = numel(subj.hdr.label{i});
            % fprintf('number of chars %d\n',nchar);
            fwrite(fid,nchar,'int32');
            fwrite(fid,subj.hdr.label{i},'char');
        end
        % number of components
        ncomp = size(subj.components,1);
        % fprintf('number of components %d\n',ncomp)
        fwrite(fid,ncomp,'int32');
        % components
        % fprintf('writing components of size %d %d\n',size(subj.components))
        fwrite(fid,subj.components,'float64');
        % number of trials
        ntrial = length(subj.trial);
        % fprintf('number of trials %d\n',ntrial)
        fwrite(fid,ntrial,'int32');
        % sample rate
        % fprintf('sample rate %d\n',subj.fsample)
        fwrite(fid,subj.fsample,'int32');
        % projected trials
        for i = 1:length(subj.trial)
            % size of trial
            proj = permute(subj.projected{i},[2 1]);
            trial_size = size(proj);
            fprintf('Trial size: %d %d\n',trial_size)
            fwrite(fid,trial_size,'int32');
            fwrite(fid,proj,'float64');
        end
    catch e
        fclose(fid);
        rethrow(e);
    end
end
