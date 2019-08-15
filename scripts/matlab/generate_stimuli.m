run(fullfile('..','..','src','matlab','util','setup.m'));
addpath(stimulus_dir);

config = [];
config.speaker_order = {'vadem','mensc','letzt'};
config.normval = 5;

% determines pitch shift
config.analysis_len = 64;
config.synthesis_len = 74;

config.target_len = 1;
config.switch_len = 1.2;
config.min_stay_len = 0.5;
config.jitter_period = 0.2;
config.min_target_start = 1.5;

test_block_cfg = [];
test_block_cfg.target_cases = [1 1; 1 2; 2 1; 2 2; -1 -1];
test_block_cfg.target_probs = [3;2;2;1;2]/10;
test_block_cfg.num_trials = 50;
config.test_block_cfg = test_block_cfg;

train_block_cfg = [];
train_block_cfg.target_cases = ...
    [1 1; -1 -1; 2 2; -1 -1; 1 2; 2 1; -1 -1; 1 1; 2 2; -1 -1];
train_block_cfg.cond_rep = 4;
train_block_cfg.target_probs = ...
    ones(size(train_block_cfg.target_cases,1),1)/...
        size(train_block_cfg.target_cases,1);
train_block_cfg.num_trials = size(train_block_cfg.target_cases,1)*...
    train_block_cfg.cond_rep;
config.train_block_cfg = train_block_cfg;


% setup and save a new mixture configuration (as config.json); each call
% creates a new set of mixtures, as this relies on a number of RNG results
stimdata_dir = redatedir(stimdata_dir);
config = configure_mixtures(stimdata_dir,config);

% This actually generates the audio, based on config.json. You can generate the
% same exact audio again by using the same config.json file. (Just re-run this
% line alone)
hrtf_file = fullfile(base_dir,'data','exp_raw','stimuli','hrtf_b_nh172.sofa');
create_mixtures(stimdata_dir,hrtf_file);
create_mixtures(fullfile(base_dir,'data','exp_pro','stimuli','2019-03-28'),...
    hrtf_file);

% TODO: auto generate description for the location of the target
