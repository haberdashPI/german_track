run('../util/setup.m')

eeg_files = dir(fullfile(raw_data_dir,'*.bdf'));
for i = 1:length(eeg_files)

    % TODO: skip already generated files and
    % warn
    eegfile = eeg_files(i).name;
    numstr = regexp(eegfile,'([0-9]+)_','tokens');
    sid = str2double(numstr{1}{1});
    result_file = fullfile(data_dir,sprintf('eeg_events_%03d.csv',sid));

    if exist(result_file,'file')
        warning('The file %s already exists. Skipping...',result_file);
        continue;
    end

    fprintf('reading events for %s\n',eegfile);
    fprintf('Found SID = %d\n',sid);
    event = ft_read_event(fullfile(raw_data_dir,eegfile));

    type = {event.type};
    status_types = zeros(length(type),1);
    for j = 1:length(type)
        status_types(j) = logical(strcmp(type(j),'STATUS'));
    end

    code = [event.value]';

    sample = [event.sample];
    sample = sample(logical(status_types))';

    SID = repmat(sid,length(code),1);
    df = table(code,sample,SID);

    writetable(df,result_file);

    head = ft_read_header(fullfile(raw_data_dir,eegfile));
    fprintf('File sample rate: %d\n',head.Fs);
end
