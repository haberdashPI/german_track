% First sanity check: can we more accurately recover sources in the right ear
% when listeners are instructed to attend to the right ear.

% wait a minute: how does this actually work,
% do we follow a single source, or just all source for a given ear?
% what is the decoder looking at? a mixture of the sources?

eeg_files = dir(fullfile(data_dir,'eeg_response*.mat'));

[eeg,events,sid] = load_subject(eeg_files(8).name);

for i = 1:length(eeg.trial)
  if strcmp(events{i,'condition'},'object')
    % right ear decoders


    % left ear decoders
