function save_subject_components(mcca,filename)
    fid = fopen(filename,'w','n','UTF-8');
    try
        % number of channels
        nchan = size(mcca.label,1);
        fwrite(fid,nchan,'int32');
        % channel names
        for i = 1:length(mcca.label)
            nchar = numel(mcca.label{i});
            fwrite(fid,nchar,'int32');
            fwrite(fid,mcca.label{i},'char');
        end
        % number of components
        if size(mcca.components,2) ~= nchan
            error("Expected each component to have %d channels.",nchan)
        end
        ncomp = size(mcca.components,1);
        fwrite(fid,ncomp,'int32');
        % components
        fwrite(fid,mcca.components,'float64');
        % number of trials
        ntrial = length(mcca.projected);
        fwrite(fid,ntrial,'int32');
        % sample rate
        fwrite(fid,mcca.hdr.Fs,'int32');
        % projected trials
        for i = 1:length(mcca.projected)
            % size of trial
            trial_size = size(mcca.projected{i});
            fwrite(fid,trial_size,'int32');
            fwrite(fid,mcca.projected{i},'float64');
        end
    catch e
        fclose(fid);
        rethrow(e);
    end
end
