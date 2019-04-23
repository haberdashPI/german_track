run('../util/setup.m')

total_times = 0;
total_channels = 0;

time_slices = {};

% STEP 1: look for any really egregious % artificats and decide how to get
% rid of them (leave most of them in, since mCCA should clean a lot of it up)

[eeg,~,~] = load_subject('eeg_response_008.mat');

plot_cfg = [];
plot_cfg.viewmode = 'vertical';
plot_cfg.preproc.detrend = 'yes';
plot_cfg.eegscale = 1000;
plot_cfg.mychan = ft_channelselection('EX*',eeg)
plot_cfg.mychanscale = 1000;

use_channels = {'A*','B*','EXG1','EXG2','EXG3','EXG4','EXG5','EXG6','-A28'};
channel_names = ft_channelselection(use_channels,eeg)
channel_indices = cellfun(@(x){ any(cellfun(@(y) strcmp(x,y),channel_names)) }, eeg.label);
channel_indices = find(cell2mat(channel_indices));
all_eeg = {}
trial_order = {}

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
n_trials = 150;
x = zeros(length(channel_indices) * length(all_eeg),n_trials*n_times);
for i = 1:length(all_eeg)
    for trial_i = 1:length(trial_order{i})
        trial = trial_order{i}(trial_i);
        trial_time_indices = 1:min(size(all_eeg{i}.trial{trial},2),n_times);
        feature_indices = (i-1)*length(channel_indices) + ...
            (1:length(channel_indices));
        time_indices = (trial_i-1)*n_times + (1:length(trial_time_indices));
        x(feature_indices,time_indices) = ...
            all_eeg{i}.trial{trial}(channel_indices,trial_time_indices);
    end
end
x = x';
x = x - repmat(mean(x,1), size(x,1),1); % subtract mean from each column
C = x'*x; % covariance matrix

nchan = length(channel_indices);
[A,score,AA] = nt_mcca(C,nchan);

comp = X * A; % MCCA components, where first column is most repeatable component across subjects
nkeep = 20; % number of components to keep

% Project out all but first "nkeep" components
for i = 1:nsub
    arr = <subject data in "tall" form, dim= ntime*ntrial x nchan>
    iA = AA{i}; % subject-specific MCCA weights
    eye_select = zeros(size(iA,2),1);
    eye_select(1:nkeep) = 1;
    y = arr* (iA*diag(eye_select)*pinv(iA));
    % y: MCCA-cleaned data for subject i
end

