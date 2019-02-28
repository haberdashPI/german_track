% First sanity check: can we more accurately recover sources in the right ear
% when listeners are instructed to attend to the right ear.

eeg_files = dir(fullfile(data_dir,'eeg_response*.mat'));

[eeg,events,sid] = load_subject(eeg_files(8).name);

for i = 1:length(eeg.trial)
  if strcmp(events{i,'condition'},'object')
    % right ear decoders


    % left ear decoders
