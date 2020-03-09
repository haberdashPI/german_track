function [trial,label] = load_subject_binary(filename)
    fid = fopen(filename,'r','n','UTF-8');
    try
        nchan = fread(fid,1,'int32');
        label = cell(nchan,1);
        for i = 1:nchan
            nchar = fread(fid,1,'int32');
            label{i} = convertCharsToStrings(native2unicode(fread(fid,nchar,'char')));
        end
        ntrial = fread(fid,1,'int32');
        sr = fread(fid,1,'int32');

        trial = cell(1,ntrial);
        for i = 1:ntrial
            trial_size = fread(fid,2,'int32')';
            trial{i} = fread(fid,trial_size,'float64');
        end
    catch e
        fclose(fid);
        rethrow(e);
    end
end
