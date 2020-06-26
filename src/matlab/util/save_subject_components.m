function save_subject_components(mcca,filename)
    try
        if exist(filename, 'file')
            delete(filename);
        end

        h5create(filename,'/channels',length(mcca.label),...
            'Datatype','string');
        h5write(filename,'/channels',string(mcca.label))

        h5create(filename,'/components',size(mcca.components),'Datatype','double');
        h5write(filename,'/components',mcca.components);

        % number of trials
        ntrials = length(mcca.projected);
        h5create(filename,'/trials/count',[1 1],'Datatype','int32');
        h5write(filename,'/trials/count',ntrials);
        h5create(filename,'/trials/samplerate',[1 1],'Datatype','double');
        h5write(filename,'/trials/samplerate',mcca.hdr.Fs);

        for i = 1:ntrials
            h5create(filename,sprintf('/trials/%03d',i),size(mcca.projected{i}),...
                'Datatype','double');
            h5write(filename,sprintf('/trials/%03d',i),mcca.projected{i});
        end
    catch e
        delete(filename);
        rethrow(e)
    end
end
