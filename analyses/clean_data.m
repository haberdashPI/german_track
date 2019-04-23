run('../util/setup.m')

total_times = 0;
total_channels = 0;

time_slices = {};

% STEP 1: look for any really egregious
% artificats and get rid of them (leave most of them in, since mCCA
% should clean a lot of it up)
[eeg,~,~] = load_subject('eeg_response_008.mat');

plot_cfg = [];
plot_cfg.viewmode = 'vertical';
plot_cfg.preproc.detrend = 'yes';
plot_cfg.eegscale = 1000;
plot_cfg.mychan = ft_channelselection('EX*',eeg)
plot_cfg.mychanscale = 1000;

use_channels = {'A*','B*','EX1','EX2','EX3','EX4','EX5','EX6','-A28'};
feature_names = ft_channelselection(use_channels,eeg)

all_eeg = {}
trial_order = {}

% thoughts: is should probably just get rid of A28
% (it's worth troubleshooting, to see if it's that particular
% selectrode, or that particular location)

% TODO: sum up the total times and channels % to assmple the data later
total_time = 0;
total_features = 0;

% visualize all the data, and make sure there isn't anything
% really egregious that we need to get rid of
% subj 8 ----------------------------------------
[eeg,stim_events,~] = load_subject('eeg_response_008.mat');
ft_databrowser(plot_cfg,eeg);
all_eeg = [all_eeg {eeg}];
trial_order = [trial_order {sort_trial_times(eeg,stim_events)}]

% subj 9 ----------------------------------------
% I think I shoud remove this participant; there's way too much noise
% (something is wrong)
[eeg,~,~] = load_subject('eeg_response_009.mat');
ft_databrowser(plot_cfg,eeg);

% subj 10 ----------------------------------------
[eeg,~,~] = load_subject('eeg_response_010.mat');
ft_databrowser(plot_cfg,eeg);
all_eeg = [all_eeg {eeg}];
trial_order = [trial_order {sort_trial_times(eeg,stim_events)}]

% subj 11 ----------------------------------------
% bad trials: 60-67
[eeg,~,~] = load_subject('eeg_response_011.mat');
ft_databrowser(plot_cfg,eeg);
all_eeg = [all_eeg {eeg}];
trial_order = [trial_order {sort_trial_times(eeg,stim_events)}]

% subj 12 ----------------------------------------
[eeg,~,~] = load_subject('eeg_response_012.mat');
ft_databrowser(plot_cfg,eeg);
all_eeg = [all_eeg {eeg}];
trial_order = [trial_order {sort_trial_times(eeg,stim_events)}]

% subj 13 ----------------------------------------
[eeg,~,~] = load_subject('eeg_response_013.mat');
ft_databrowser(plot_cfg,eeg);
all_eeg = [all_eeg {eeg}];
trial_order = [trial_order {sort_trial_times(eeg,stim_events)}]

% subj 14 ----------------------------------------
[eeg,~,~] = load_subject('eeg_response_014.mat');
ft_databrowser(plot_cfg,eeg);
all_eeg = [all_eeg {eeg}];
trial_order = [trial_order {sort_trial_times(eeg,stim_events)}]

n_times = 10*64;
n_trials = 150
x = zeros(length(use_channels) * length(all_eeg),n_trials*n_times);
for i = 1:length(all_eeg)
    for trial = trial_order{i};
        feature_indices = (i-1)*length(use_channels) + (1:length(use_channels))
        time_indices = (trial-1)*n_times + 1:n_times
        % TODO: trial_time_indices and trial_chan_indices
        x(feature_indices,time_indices) = ...
            all_eeg{i}.trial{trial}(trial_time_indices,trial_chan_indices)
    end
end
% TODO: STOPPED HERE
% stimuli are shuffled: re-order them so stimuli match across participants


% STEP 2: assemble the matrices
total_times = 0;
total_channels = length(use_channels) * length(trials)
for i = 1:length(all_eeg)
    eeg = all_eeg{i};
    total_times = total_times + sum(cellfun(@(x) size(x,2),eeg.trial));
end

% TODO: arrange all trials by

% WARNING: nt_mcca assumes same number of channels per subject (if we
% remove a channel in one subject but not others, we have to interpolate)
x = zeros(total_channels,total_times)
for i = 1:length(all_eeg)
    x()
