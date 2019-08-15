function save_subject(dat,file)
    global data_dir

    eegfile = fullfile(data_dir,file);
    save(eegfile,'dat')
end
