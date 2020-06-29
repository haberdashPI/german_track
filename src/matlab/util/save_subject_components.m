function save_subject_components(mcca,filename)
    nchans = size(mcca.trial{1},1);
    h5create(filename,'/channels',[length(mcca.label) 1],...
        'DataType','string');
    h5write(filename,'/channels',string(mcca.label))

    h5create(filename,'/components',size(mcca.components),'DataType','double');
    h5write(filename,'/components',mcca.components);

    % number of trials
    ntrial = length(mcca.projected);
    h5create(filename,'/trials/count',[1 1],'DataType','int32');
    h5write(filename,'/trials/count',ntrial);
    h5create(filename,'/trials/samplerate',[1 1],'DateType','double');
    h5write(filename,'/trials/samplerate',mcca.hdr.Fs);

    for i = 1:ntrials
        h5create(filename,sprintf('/trials/%03d',i),size(mcca.trial{i}),...
            'DataType','double');
        h5write(filename,sprintf('/trials/%03d',i),mcca.projected{i});
    end
end
