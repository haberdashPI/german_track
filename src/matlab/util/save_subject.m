function save_subject(dat,file)
    global processed_datadir

    eegfile = fullfile(processed_datadir, 'eeg', file);
    save(eegfile,'dat')
end
