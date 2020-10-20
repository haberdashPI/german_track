% setup environment
run(fullfile('src','matlab','util','setup.m'));

% This script is for extracting data from the behavioral expeirment
% (that occured before eeg subjects)

% convert data from mat file to an easy to read CSV file
meta = importdata(fullfile(raw_datadir, '..', 'behavioral', 'experiment_record.mat'));
trial = (1:50)';
speaker = meta.experiment_cfg.test_block_cfg.trial_dev_speakers;
direction = meta.experiment_cfg.test_block_cfg.trial_dev_direction;
target_time = meta.experiment_cfg.test_block_cfg.target_times;

writetable(table(trial, speaker, direction, target_time), ...
    fullfile(processed_datadir,'behavioral','stimuli','stim_info.csv'));
