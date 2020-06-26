function [eeg_data,stim_events,sid] = load_subject(file)
    global processed_datadir

    eegfile = fullfile(processed_datadir, 'eeg', file);
    eegfiledata = load(eegfile);
    eeg_data = eegfiledata.dat;

    numstr = regexp(eegfile,'([0-9]+)(_[a-z_]+)?.mat','tokens');
    sid = str2num(numstr{1}{1});
    eventfile = fullfile(processed_datadir, 'eeg', sprintf('sound_events_%03d.csv',sid));
    stim_events = readtable(eventfile);
end
