datadir = '/Users/davidlittle/Data/EEGAttn_David_Little_2018_01_24/'
eegfile = fullfile(datadir,'2018-01-24_0001_DavidLittle_record.bdf')
ft_defaults

event = ft_read_event(eegfile)


type = {event.type};
status_types = zeros(length(type),1);
for i = 1:length(type)
  status_types(i) = logical(strcmp(type(i),"STATUS"));
end

code = [event.value]';

sample = [event.sample];
sample = sample(logical(status_types))'

df = table(code,sample)

writetable(df,'eeg_events.csv')

head = ft_read_header(eegfile)
disp(head.Fs)
