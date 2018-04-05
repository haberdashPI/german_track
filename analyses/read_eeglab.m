% for now this is just for sid = 2

run('../util/setup.m')

eegfiles = dir(fullfile(raw_data_dir,'*_ICA.set'));
sid = 2;
result_file = fullfile(data_dir,sprintf('eeg_response_%03d.bdf.mat',sid));

eegfile = eegfiles(1).name
header = ft_read_header(fullfile(raw_data_dir,eegfile));

cfg = [];
cfg.datafile = fullfile(raw_data_dir,eegfile);
cfg.headerfile = fullfile(raw_data_dir,eegfile);
eeg_data = ft_preprocessing(cfg);

save(result_file,'eeg_data')