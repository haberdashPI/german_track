run('../util/setup.m')

eeg_files = dir(fullfile(raw_data_dir,'*.bdf'));
for i = 1:length(eeg_files) % TEMPORARY, normally 1:length...

  % TODO: skip already generated files and
  % warn
  eegfile = eeg_files(i).name;
  numstr = regexp(eegfile,'_([0-9]+)_','tokens');
  sid = str2num(numstr{1}{1});
  result_file = fullfile(data_dir,sprintf('eeg_events_%04d.csv',sid));

  if exists(result_file)
    warn(['The file ' result_file ' already exists. Skipping...']);
    continue;
  end

  disp(['reading events for ' eegfile]);
  disp(['Found SID = ' num2str(sid)]);
  event = ft_read_event(fullfile(raw_data_dir,eegfile));

  type = {event.type};
  status_types = zeros(length(type),1);
  for i = 1:length(type)
    status_types(i) = logical(strcmp(type(i),"STATUS"));
  end

  code = [event.value]';

  sample = [event.sample];
  sample = sample(logical(status_types))';

  SID = repmat(sid,length(code),1);
  df = table(code,sample,SID);

  writetable(df,);

  head = ft_read_header(fullfile(raw_data_dir,eegfile));
  disp(['File sample rate: ' num2str(head.Fs)]);
end
