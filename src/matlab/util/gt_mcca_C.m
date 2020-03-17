function C = gt_mcca_C(files,maxlen,conds,stimuli,channels)
    global cache_dir;
    global data_dir;

    nchans = length(channels);
    nsubj = length(files);
    C = zeros(nchans*nsubj);

    textprogressbar('Computing C...');
    onCleanup(@() textprogressbar(''));

    totalitr = length(stimuli)*length(conds)*length(files);
    n = 0;
    for t = stimuli
        for c = 1:length(conds)

            x = zeros(maxlen,nchans*nsubj);
            wt = zeros(maxlen,nchans*nsubj);

            for i = 1:length(files)
                n = n+1;

                % load eeg file
                filename = files(i).name;
                filepath = fullfile(cache_dir,'eeg',filename);
                [eeg,~,w] = load_subject_binary(filepath);

                numstr = regexp(filepath,'([0-9]+)_','tokens');
                sid = str2num(numstr{1}{1});

                %% read in the events
                event_file = fullfile(data_dir,sprintf('sound_events_%03d.csv',sid));
                stim_events = readtable(event_file);

                %% find the index for this stimulus index and condition
                row = find(strcmp(stim_events.condition,conds{c}) & stim_events.sound_index == t);
                if isempty(row)
                    warning("Missing stimulus %d in condition '%s' of sid %d",t,conds{c},sid);
                    continue
                end
                if length(row) > 1
                    error("Found multiple rows??")
                end

                data = eeg{row}(channels,:);
                wdata = w{row}(channels,:);
                x(1:size(data,2),(1:nchans) + (i-1)*nchans) = (data.*wdata)';
                wt(1:size(data,2),(1:nchans) + (i-1)*nchans) = wdata';

                textprogressbar(100*n/totalitr);
            end

            % wt has 1 and 0 entries, so we can square it to
            % count observations for each x*y correlation pair
            w2 = wt'*wt;
            w2(w2 == 0) = 1; % avoid NaNs
            C = C + (x'*x)./w2;
        end
    end
end

