run('../util/setup.m')

% ======================================================================
% STEP 0: configuration

% load first file, to get channel names
[eeg,~,~] = load_subject('eeg_response_008.mat');

% pre-cleaning plot configuration
plot_cfg = [];
plot_cfg.viewmode = 'vertical';
plot_cfg.preproc.detrend = 'yes';
plot_cfg.eegscale = 1;
plot_cfg.mychan = ft_channelselection('EX*',eeg);
plot_cfg.mychanscale = 1;
plot_cfg.ylim = [-20 20];

% electrode geometry
elec = ft_read_sens(fullfile(data_dir,'biosemi64.txt'));
cfg = [];
cfg.elec = elec;
lay = ft_prepare_layout(cfg);
ft_layoutplot(cfg)

all_eeg = {}
trial_order = {}

% visual rejection plot
reject_cfg          = [];
reject_cfg.method   = 'summary';
reject_cfg.alim     = 1e-12;

% channel repair, via interpolation
channel_repair_cfg = [];
channel_repair_cfg.method = 'spline';
channel_repair_cfg.elec = elec;

% post-detrending plot configuration
plot_detrend_cfg = plot_cfg;
plot_detrend_cfg.preproc.detrend = 'no';
plot_detrend_cfg.preproc.demean = 'no';
plot_detrend_cfg.eegscale = 50;
plot_detrend_cfg.mychan = ft_channelselection('EX*',eeg)
plot_detrend_cfg.mychanscale = 50;

% ======================================================================
% STEP 1: clean the data: remove egregious artifacts, and de-trend the
% data

% subj 8 ----------------------------------------
[eeg,stim_events,~] = load_subject('eeg_response_008.mat');
ft_rejectvisual(reject_cfg,eeg);

bad_trials = [
    19
    35
    69
    117
    118
];

% zero trials, interpolate channels
eeg = zero_trials(eeg,bad_trials);
channel_repair_cfg.badchannel = {'A28'};
channel_repair_cfg.trials = 'all';
eeg = my_channelrepair(channel_repair_cfg,eeg);

% detrend the data
eeg = my_detrend(eeg,bad_trials);
alert()

% verify the result
ft_databrowser(plot_detrend_cfg,eeg);

% add to data set
all_eeg{1} = eeg;

% subj 9 ----------------------------------------
% NOTE: I may need to remove this participant
% it's quite noisey data
[eeg,stim_events,~] = load_subject('eeg_response_009.mat');
ft_rejectvisual(reject_cfg,eeg);

bad_trials = [
    18
    42
    50
    100
    101
    103
    116
    117
    118
    134
    135
    137
    138
    139
    140
    142
    147
];

% zero trials, interpolate channels
eeg = zero_trials(eeg,bad_trials);
channel_repair_cfg.badchannel = {'A1','A2','A7','B1','B2','B3','B4','B21','A28',''};
channel_repair_cfg.trials = 'all';
eeg = my_channelrepair(channel_repair_cfg,eeg);

% detrend the data
eeg = my_detrend(eeg,bad_trials);
alert()

% verify the result
ft_databrowser(plot_detrend_cfg,eeg);

% add to data set
all_eeg{2} = eeg;

% subj 10 ----------------------------------------
[eeg,stim_events,~] = load_subject('eeg_response_010.mat');
ft_rejectvisual(reject_cfg,eeg);

bad_trials = [
    34
    118
];

% zero trials, interpolate channels
eeg = zero_trials(eeg,bad_trials);
channel_repair_cfg.badchannel = {'A4','A5','A28','B25'};
channel_repair_cfg.trials = 'all';
eeg = my_channelrepair(channel_repair_cfg,eeg);

% detrend the data
eeg = my_detrend(eeg,bad_trials);
alert()

% verify the result
ft_databrowser(plot_detrend_cfg,eeg);

% TODO: what's with the differetn scales for different subjects?

% add to data set
all_eeg{3} = eeg;

% subj 11 ----------------------------------------
[eeg,stim_events,~] = load_subject('eeg_response_011.mat');
ft_rejectvisual(reject_cfg,eeg);

bad_trials = [
    34
    35
    36
    39
    45
    55
    59
    60
    61
    62
    63
    64
    65
    66
    67
    68
    69
    74
    81
    83
    84
    94
    116
    124
    129
    134
    149
];

% zero trials, interpolate channels
eeg = zero_trials(eeg,bad_trials);
channel_repair_cfg.badchannel = {'A28','B31'};
channel_repair_cfg.trials = 'all';
eeg = my_channelrepair(channel_repair_cfg,eeg);

% detrend the data
eeg = my_detrend(eeg,bad_trials);
alert()

% verify the result
ft_databrowser(plot_detrend_cfg,eeg);

% add to data set
all_eeg{4} = eeg;

% subj 12 ----------------------------------------
[eeg,stim_events,~] = load_subject('eeg_response_012.mat');
ft_rejectvisual(reject_cfg,eeg);
ft_databrowser(plot_cfg,eeg);

channel_repair_cfg.badchannel = {'B26'};
channel_repair_cfg.trials = 90;
eeg = my_channelrepair(channel_repair_cfg,eeg);

channel_repair_cfg.badchannel = {'A21'};
channel_repair_cfg.trials = 102;
eeg = my_channelrepair(channel_repair_cfg,eeg);

ft_rejectvisual(reject_cfg,eeg);
ft_databrowser(plot_cfg,eeg);

% TODO: reject trials and channels

bad_trials = [
    115
];

% zero trials, interpolate channels
eeg = zero_trials(eeg,bad_trials);
% TODO:
channel_repair_cfg.badchannel = {'B26'};
channel_repair_cfg.trials = 'all';
eeg = my_channelrepair(channel_repair_cfg,eeg);


% detrend the data
eeg = my_detrend(eeg,bad_trials);
alert()

% verify the result
ft_databrowser(plot_detrend_cfg,eeg);

% add to data set
all_eeg{5} = eeg;

% subj 13 ----------------------------------------
[eeg,stim_events,~] = load_subject('eeg_response_013.mat');
ft_rejectvisual(reject_cfg,eeg);
ft_databrowser(plot_cfg,eeg);

bad_trials = [
    114
    150
];

% zero trials, interpolate channels
eeg = zero_trials(eeg,bad_trials);
channel_repair_cfg.badchannel = {'A28'};
channel_repair_cfg.trials = 'all';
eeg = my_channelrepair(channel_repair_cfg,eeg);

% detrend the data
eeg = my_detrend(eeg,bad_trials);
alert()

% verify the result
ft_databrowser(plot_detrend_cfg,eeg);

% add to data set
all_eeg{6} = eeg;

% subj 14 ----------------------------------------
[eeg,stim_events,~] = load_subject('eeg_response_014.mat');

ft_rejectvisual(reject_cfg,eeg);
% trial 17 & 35
ft_databrowser(plot_cfg,eeg);
% 17: B1, B2, A1
% 35: B26

channel_repair_cfg.badchannel = {'B1','B2','A1'};
channel_repair_cfg.trials = 17;
eeg = my_channelrepair(channel_repair_cfg,eeg);

channel_repair_cfg.badchannel = {'B26'};
channel_repair_cfg.trials = 35;
eeg = my_channelrepair(channel_repair_cfg,eeg);

ft_rejectvisual(reject_cfg,eeg);

bad_trials = [
    17
];

% zero trials, interpolate channels
eeg = zero_trials(eeg,bad_trials);
channel_repair_cfg.badchannel = {'B1','B2','B14'};
channel_repair_cfg.trials = 'all';
eeg = my_channelrepair(channel_repair_cfg,eeg);

% detrend the data
eeg = my_detrend(eeg,bad_trials);
alert()

% verify the result
ft_databrowser(plot_detrend_cfg,eeg);

% add to data set
all_eeg{7} = eeg;

% % STEP 3: use mCCA to identify shared components

n_times = 6*64;
n_trials = 150;
n_chans = length(channel_indices);
x = zeros(n_trials*n_times,n_chans * length(all_eeg));
for i = 1:length(all_eeg)
    for trial_i = 1:length(trial_order{i})
        trial = trial_order{i}(trial_i);
        trial_time_indices = 1:min(size(all_eeg{i}.trial{trial},2),n_times);
        feature_indices = (i-1)*n_chans + ...
            (1:n_chans);
        time_indices = (trial_i-1)*n_times + (1:length(trial_time_indices));
        x(time_indices,feature_indices) = ...
            all_eeg{i}.trial{trial}(channel_indices,trial_time_indices)';
    end
end
chan_mean = mean(x,1);
x = x - chan_mean; % subtract mean from each column
C = x'*x; % covariance matrix

[A,score,AA] = nt_mcca(C,n_chans);

bar(score(1:200));
nkeep = 2; % number of components to keep

% Project out all but first "nkeep" components
for i = 1:length(all_eeg)
    iA = AA{i}; % subject-specific MCCA weights
    selection = zeros(size(iA,2),1);
    selection(1:nkeep) = 1;
    all_eeg{i}.old_trial = {};
    for t = 1:length(all_eeg{i}.trial)
        all_eeg{i}.old_trial{t} = all_eeg{i}.trial{t};
        arr = all_eeg{i}.trial{t};
        proj_arr = arr(channel_indices,:)';
        mu = chan_means((i-1)*n_chans + (1:n_chans));
        proj_arr = proj_arr - mu;
        proj_arr = proj_arr * (iA*diag(selection)*pinv(iA));
        arr(channel_indices,:) = (proj_arr + mu)';
        arr(setdiff(1:end,channel_indices),:) = 0;
        all_eeg{i}.trial{t} = arr;
    end
end

ft_databrowser(plot_cfg,all_eeg{1});

% TODO: save these data and see if this "cleaned" result
% works any better (not super convinced it will, given that only 2
% components were found)

for i = 1:length(all_eeg)

