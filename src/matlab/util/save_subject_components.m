function save_subject_components(trial,components,label,weights,fs,filename)
    fid = fopen(filename,'w','n','UTF-8');
    try
        % number of channels
        nchan = size(trial{1},1);
        fwrite(fid,nchan,'int32');
        % channel names
        for i = 1:size(trial{1},1)
            nchar = numel(subj.hdr.label{i});
            fwrite(fid,nchar,'int32');
            fwrite(fid,label{i},'char');
        end
        % number of components
        ncomp = size(components,1);
        fwrite(fid,ncomp,'int32');
        % components
        fwrite(fid,components,'float64');
        % number of trials
        ntrial = length(trial);
        fwrite(fid,ntrial,'int32');
        % sample rate
        fwrite(fid,fs,'int32');
        % projected trials
        for i = 1:length(trial)
            % size of trial
            trial_size = size(trial{i});
            fwrite(fid,trial_size,'int32');
            fwrite(fid,trial{i},'float64');
        end
    catch e
        fclose(fid);
        rethrow(e);
    end
end
