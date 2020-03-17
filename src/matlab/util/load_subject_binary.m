function [trial,label,w] = load_subject_binary(filename)
    fid = fopen(filename,'r','n','UTF-8');
    try
        filever = fread(fid,1,'int32');
        if filever == 2
            hasweights = fread(fid,1,'uint8');
        else
            error("Unexpected file version number: %d",filever)
        end
        nchan = fread(fid,1,'int32');
        label = cell(nchan,1);
        for i = 1:nchan
            nchar = fread(fid,1,'int32');
            label{i} = native2unicode(fread(fid,nchar,'char')');
        end
        ntrial = fread(fid,1,'int32');
        sr = fread(fid,1,'int32');

        trial = cell(1,ntrial);
        if hasweights
            w = cell(1,ntrial);
        end
        for i = 1:ntrial
            trial_size = fread(fid,2,'int32')';
            trial{i} = fread(fid,trial_size,'float64');
            if hasweights
                w{i} = fread(fid,trial_size,'float64');
            end
        end
    catch e
        fclose(fid);
        rethrow(e);
    end
    fclose(fid);
end
