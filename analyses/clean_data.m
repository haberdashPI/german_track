run('../util/setup.m')

% STEP 1: look for any really egregious
% artificats and get rid of them (leave most of them in, since mCCA
% should clean a lot of it up)

plot_cfg = [];
plot_cfg.viewmode = 'vertical';
plot_cfg.preproc.detrend = 'yes';
plot_cfg.eegscale = 1000;
plot_cfg.mychan = ft_channelselection('EX*',eeg)
plot_cfg.mychanscale = 1000;

% thoughts: is hould probably just get rid of A28
% (it's worth troubleshooting, to see if it's that particular
% selectrode, or that particular location)

% TODO: sum up the total times and channels % to assmple the data later
total_time = 0

% subj 8
[eeg,~,~] = load_subject('eeg_response_008.mat');
ft_databrowser(plot_cfg,eeg);
total_time = sum()

% remove this participant; there's way too much noise
% (something is wrong)
[eeg,~,~] = load_subject('eeg_response_009.mat');
ft_databrowser(plot_cfg,eeg);

[eeg,~,~] = load_subject('eeg_response_010.mat');
ft_databrowser(plot_cfg,eeg);

% bad trials: 60-67
[eeg,~,~] = load_subject('eeg_response_011.mat');
ft_databrowser(plot_cfg,eeg);

[eeg,~,~] = load_subject('eeg_response_012.mat');
ft_databrowser(plot_cfg,eeg);

[eeg,~,~] = load_subject('eeg_response_013.mat');
ft_databrowser(plot_cfg,eeg);

[eeg,~,~] = load_subject('eeg_response_014.mat');
ft_databrowser(plot_cfg,eeg);

% STEP 2: assemble the matrices

