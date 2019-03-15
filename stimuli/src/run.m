% This is the top-level script used to generate stimuli

run('../../util/setup.m')
path('.')

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

train_block_cfg = [];
train_block_cfg.target_cases = ...
    [1 1; -1 -1; 2 2; -1 -1; 1 2; 2 1; -1 -1; 1 1; 2 2; -1 -1];
train_block_cfg.cond_rep = 4;
train_block_cfg.target_probs = ...
    ones(size(train_block_cfg.target_cases,1),1)/...
        size(train_block_cfg.target_cases,1);
train_block_cfg.num_trials = size(train_block_cfg.target_cases,1)*...
    train_block_cfg.cond_rep;

config.hrtf_file = fullfile(base_dir,'stimuli','hrtfs','hrtf_b_nh172.sofa')

% setup and save mixture configuration
config = configure_mixtures(fullfile(base_dir,'stimuli'),config);

% NOTE: you can load `config` at a later date, and generate the same
% audio use `create_mixtures`.

create_mixtures(config);

% TODO: auto generate description for the location of the target
