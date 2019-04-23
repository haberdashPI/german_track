run('../util/setup.m')

total_times = 0;
total_channels = 0;

time_slices = {};

% STEP 1: clean the data: remove egregious artifacts, and de-trend the
% data

[eeg,~,~] = load_subject('eeg_response_008.mat');

plot_cfg = [];
plot_cfg.viewmode = 'vertical';
plot_cfg.preproc.detrend = 'yes';
% plot_cfg.preproc.demean = 'yes';
plot_cfg.eegscale = 200;
plot_cfg.mychan = ft_channelselection('EX*',eeg)
plot_cfg.mychanscale = 200;

use_channels = {'A*','B*','EXG1','EXG2','EXG3','EXG4','EXG5','EXG6','-A28'};
channel_names = ft_channelselection(use_channels,eeg)
channel_indices = cellfun(@(x){ any(cellfun(@(y) strcmp(x,y),channel_names)) }, eeg.label);
channel_indices = find(cell2mat(channel_indices));

plot_cfg.channel = channel_names;

all_eeg = {}
trial_order = {}

% subj 8 ----------------------------------------
[eeg,stim_events,~] = load_subject('eeg_response_008.mat');
% ft_databrowser(plot_cfg,eeg);
trial_order = [trial_order {sort_trial_times(eeg,stim_events)}];
bad_trials = [
    69
    118
]';

% zero out bad trials and channels
% zero out all data 500ms past the stimulus
% TODO: don't zero out the times, since that is dealt
% with in `read_eeg_response` now.
config = read_json(fullfile(stimulus_dir,"config.json"));
bad_channels = setdiff(1:size(eeg.trial{1}),channel_indices);
for i = 1:length(eeg.trial)
    if any(i == bad_trials)
        eeg.trial{i}(:,:) = 0;
    else
        eeg.trial{i}(bad_channels,:) = 0;
        trial_file = fullfile(stimulus_dir,"mixtures","testing",...
            sprintf("trial_%02d.wav",stim_events.sound_index(i)));
        [trial_audio,fs] = audioread(trial_file);
        audio_samples = round(size(trial_audio,1)/fs*eeg.fsample);
        len = min(audio_samples + round(0.5*eeg.fsample),...
            size(eeg.trial{i},2));

        eeg.trial_end(i) = len;
        eeg.trial{i} = eeg.trial{i}(:,1:len);
        eeg.time{i} = (1:len) ./ eeg.fsample;
    end
end

% linear detrending
for i = 1:length(eeg.trial)
    time_indices = 1:eeg.trial_end(i);
    data = eeg.trial{i}(channel_indices,time_indices);
    if ~any(i == bad_trials)
        eeg.trial{i}(channel_indices,time_indices) = nt_demean(data);
        % eeg.trial{i}(channel_indices,time_indices) = nt_detrend(data,1);
    end
end

ft_databrowser(plot_cfg,eeg);

all_eeg = [all_eeg {eeg}];

% subj 9 ----------------------------------------
% I think I shoud remove this participant; there's way too much noise
% (something is wrong)
% [eeg,stim_events,~] = load_subject('eeg_response_009.mat');
% % ft_databrowser(plot_cfg,eeg);
% all_eeg = [all_eeg {eeg}];
% trial_order = [trial_order {sort_trial_times(eeg,stim_events)}];

% subj 10 ----------------------------------------
[eeg,stim_events,~] = load_subject('eeg_response_010.mat');
% ft_databrowser(plot_cfg,eeg);
all_eeg = [all_eeg {eeg}];
trial_order = [trial_order {sort_trial_times(eeg,stim_events)}];

% subj 11 ----------------------------------------
% bad trials: 60-67
[eeg,stim_events,~] = load_subject('eeg_response_011.mat');
% ft_databrowser(plot_cfg,eeg);
all_eeg = [all_eeg {eeg}];
trial_order = [trial_order {sort_trial_times(eeg,stim_events)}];

% subj 12 ----------------------------------------
[eeg,stim_events,~] = load_subject('eeg_response_012.mat');
% ft_databrowser(plot_cfg,eeg);
all_eeg = [all_eeg {eeg}];
trial_order = [trial_order {sort_trial_times(eeg,stim_events)}];

% subj 13 ----------------------------------------
[eeg,stim_events,~] = load_subject('eeg_response_013.mat');
% ft_databrowser(plot_cfg,eeg);
all_eeg = [all_eeg {eeg}];
trial_order = [trial_order {sort_trial_times(eeg,stim_events)}];

% subj 14 ----------------------------------------
[eeg,stim_events,~] = load_subject('eeg_response_014.mat');
% ft_databrowser(plot_cfg,eeg);
all_eeg = [all_eeg {eeg}];
trial_order = [trial_order {sort_trial_times(eeg,stim_events)}];

muscle_cfg = [];
muscle_cfg.artfctdef.muscle.channel = ft_channelselection({'EX*'},eeg);
[cfg,artifact] = ft_artifact_muscle(muscle_cfg,eeg);


% STEP 2: use mCCA to identify shared components

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

