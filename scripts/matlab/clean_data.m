run(fullfile('..','..','src','matlab','util','setup.m'));

% ======================================================================
% configuration

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
elec = ft_read_sens(fullfile(raw_data_dir,'biosemi64.txt'));
cfg = [];
cfg.elec = elec;
lay = ft_prepare_layout(cfg);
ft_layoutplot(cfg)

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
plot_detrend_cfg.mychan = ft_channelselection('EX*',eeg);
plot_detrend_cfg.mychanscale = 50;

% ======================================================================
% clean the data: remove egregious artifacts, and de-trend the
% data

% subj 8 ----------------------------------------
[eeg,stim_events,sid] = load_subject('eeg_response_008.mat');

% ft_rejectvisual(reject_cfg,eeg);

bad_trials = [
    19
    35
    69
    117
    118
];
stim_events.bad_trial = zeros(size(stim_events,1),1);
stim_events(bad_trials,'bad_trial') = num2cell(ones(length(bad_trials),1));
writetable(stim_events,fullfile(data_dir,sprintf('sound_events_%03d.csv',sid)))

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
% trial_order{1} = sort_trial_times(eeg,stim_events);

save_subject(eeg,'eeg_response_008_cleaned.mat');

% subj 9 ----------------------------------------
% NOTE: I may need to remove this participant
% it's quite noisey data
[eeg,stim_events,sid] = load_subject('eeg_response_009.mat');
% ft_rejectvisual(reject_cfg,eeg);

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
stim_events.bad_trial = zeros(size(stim_events,1),1);
stim_events(bad_trials,'bad_trial') = num2cell(ones(length(bad_trials),1));
writetable(stim_events,fullfile(data_dir,sprintf('sound_events_%03d.csv',sid)))

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
% trial_order{2} = sort_trial_times(eeg,stim_events);
save_subject(eeg,'eeg_response_009_cleaned.mat')

% subj 10 ----------------------------------------
[eeg,stim_events,sid] = load_subject('eeg_response_010.mat');
% ft_rejectvisual(reject_cfg,eeg);

bad_trials = [
    34
    118
];
stim_events.bad_trial = zeros(size(stim_events,1),1);
stim_events(bad_trials,'bad_trial') = num2cell(ones(length(bad_trials),1));
writetable(stim_events,fullfile(data_dir,sprintf('sound_events_%03d.csv',sid)))

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
% trial_order{3} = sort_trial_times(eeg,stim_events);
save_subject(eeg,'eeg_response_010_cleaned.mat')

% subj 11 ----------------------------------------
[eeg,stim_events,sid] = load_subject('eeg_response_011.mat');
% ft_databrowser(plot_cfg,eeg);
% ft_rejectvisual(reject_cfg,eeg);

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
stim_events.bad_trial = zeros(size(stim_events,1),1);
stim_events(bad_trials,'bad_trial') = num2cell(ones(length(bad_trials),1));
writetable(stim_events,fullfile(data_dir,sprintf('sound_events_%03d.csv',sid)))

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
% trial_order{4} = sort_trial_times(eeg,stim_events);
save_subject(eeg,'eeg_response_011_cleaned.mat')

% TODO: stopped here
% subj 12 ----------------------------------------
[eeg,stim_events,sid] = load_subject('eeg_response_012.mat');
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

bad_trials = [
    115
];
stim_events.bad_trial = zeros(size(stim_events,1),1);
stim_events(bad_trials,'bad_trial') = num2cell(ones(length(bad_trials),1));
writetable(stim_events,fullfile(data_dir,sprintf('sound_events_%03d.csv',sid)))

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
% trial_order{5} = sort_trial_times(eeg,stim_events);
save_subject(eeg,'eeg_response_012_cleaned.mat')

% subj 13 ----------------------------------------
[eeg,stim_events,sid] = load_subject('eeg_response_013.mat');
ft_rejectvisual(reject_cfg,eeg);
ft_databrowser(plot_cfg,eeg);

bad_trials = [
    114
    150
];
stim_events.bad_trial = zeros(size(stim_events,1),1);
stim_events(bad_trials,'bad_trial') = num2cell(ones(length(bad_trials),1));
writetable(stim_events,fullfile(data_dir,sprintf('sound_events_%03d.csv',sid)))

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
% trial_order{6} = sort_trial_times(eeg,stim_events);
save_subject(eeg,'eeg_response_013_cleaned.mat')

% subj 14 ----------------------------------------
[eeg,stim_events,sid] = load_subject('eeg_response_014.mat');

ft_rejectvisual(reject_cfg,eeg);
% trial 17 & 35
ft_databrowser(plot_cfg,eeg);
% 17: B1, B2, A1
% 35: B26

% channel_repair_cfg.badchannel = {'B1','B2','A1'};
% channel_repair_cfg.trials = 17;
% eeg = my_channelrepair(channel_repair_cfg,eeg);

channel_repair_cfg.badchannel = {'B26'};
channel_repair_cfg.trials = 35;
eeg = my_channelrepair(channel_repair_cfg,eeg);

ft_rejectvisual(reject_cfg,eeg);

bad_trials = [
    17
];
stim_events.bad_trial = zeros(size(stim_events,1),1);
stim_events(bad_trials,'bad_trial') = num2cell(ones(length(bad_trials),1));
writetable(stim_events,fullfile(data_dir,sprintf('sound_events_%03d.csv',sid)))

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
% trial_order{7} = sort_trial_times(eeg,stim_events);
save_subject(eeg,'eeg_response_014_cleaned.mat');

% subj 16 ----------------------------------------

[eeg,stim_events,sid] = load_subject('eeg_response_016.mat');

ft_rejectvisual(reject_cfg,eeg);

bad_trials = [
    69,
    82,
    84,
    85,
    86,
    117
];
stim_events.bad_trial = zeros(size(stim_events,1),1);
stim_events(bad_trials,'bad_trial') = num2cell(ones(length(bad_trials),1));
writetable(stim_events,fullfile(data_dir,sprintf('sound_events_%03d.csv',sid)))

eeg = zero_trials(eeg,bad_trials);
channel_repair_cfg.badchannel = {'A4','B3','A28'}
channel_repair_cfg.trials = 'all';
eeg = my_channelrepair(channel_repair_cfg,eeg);

% detrend the data
eeg = my_detrend(eeg,bad_trials);
alert()

% verify the result
ft_databrowser(plot_detrend_cfg,eeg);

save_subject(eeg,'eeg_response_016_cleaned.mat');

% subj 17 ----------------------------------------
[eeg,stim_events,sid] = load_subject('eeg_response_017.mat');

ft_rejectvisual(reject_cfg,eeg);
ft_databrowser(plot_cfg,eeg);

bad_trials = [
    17,
    67,
    84
];

stim_events.bad_trial = zeros(size(stim_events,1),1);
stim_events(bad_trials,'bad_trial') = num2cell(ones(length(bad_trials),1));
writetable(stim_events,fullfile(data_dir,sprintf('sound_events_%03d.csv',sid)))

eeg = zero_trials(eeg,bad_trials);
channel_repair_cfg.badchannel = {'A28'};
channel_repair_cfg.trials = 'all';
eeg = my_channelrepair(channel_repair_cfg,eeg);

eeg = my_detrend(eeg,bad_trials);
alert()

% verify the result
ft_databrowser(plot_detrend_cfg,eeg);

save_subject(eeg,'eeg_response_017_cleaned.mat');

% subj 18 ----------------------------------------
[eeg,stim_events,sid] = load_subject('eeg_response_018.mat');

ft_rejectvisual(reject_cfg,eeg);
ft_databrowser(plot_cfg,eeg);

bad_trials = [
    103
];

stim_events.bad_trial = zeros(size(stim_events,1),1);
stim_events(bad_trials,'bad_trial') = num2cell(ones(length(bad_trials),1));
writetable(stim_events,fullfile(data_dir,sprintf('sound_events_%03d.csv',sid)))

eeg = zero_trials(eeg,bad_trials);
channel_repair_cfg.badchannel = {'A24','B3'}
channel_repair_cfg.trials = 'all';
eeg = my_channelrepair(channel_repair_cfg,eeg);

eeg = my_detrend(eeg,bad_trials);
alert()

% verify the result
ft_databrowser(plot_detrend_cfg,eeg);

save_subject(eeg,'eeg_response_018_cleaned.mat');

% subj 19 ----------------------------------------
[eeg,stim_events,sid] = load_subject('eeg_response_019.mat');

ft_rejectvisual(reject_cfg,eeg);
ft_databrowser(plot_cfg,eeg);

bad_trials = [
    106
];

stim_events.bad_trial = zeros(size(stim_events,1),1);
stim_events(bad_trials,'bad_trial') = num2cell(ones(length(bad_trials),1));
writetable(stim_events,fullfile(data_dir,sprintf('sound_events_%03d.csv',sid)))

eeg = zero_trials(eeg,bad_trials);
channel_repair_cfg.badchannel = {'B25','A8','A28'};
channel_repair_cfg.trials = 'all';
eeg = my_channelrepair(channel_repair_cfg,eeg);

% double check rejections
ft_rejectvisual(reject_cfg,eeg);

eeg = my_detrend(eeg,bad_trials);
alert()

% verify the result
ft_databrowser(plot_detrend_cfg,eeg);

% NOTE: this participant is pretty noisy... lots of what look to be muscle
% artifacts

save_subject(eeg,'eeg_response_019_cleaned.mat');
