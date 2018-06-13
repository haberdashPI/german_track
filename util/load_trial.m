function [eeg_data,stim_events,sid] = load_trial(file)
  global data_dir

  eegfile = fullfile(data_dir,file)
  eegfiledata = load(eegfile);
  eeg_data = eegfiledata.dat;

  numstr = regexp(eegfile,'_([0-9]+)(_ica)?.bdf','tokens');
  sid = str2num(numstr{1}{1});
  eventfile = fullfile(data_dir,sprintf('sound_events_%03d.csv',sid));
  stim_events = readtable(eventfile);
end
