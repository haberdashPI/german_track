run('../util/setup.m')

% ======================================================================
% STEP 1: load cleaned data
files = dir(fullfile(data_dir,'*_cleaned.mat'));

all_eeg = {};
trial_order = {};
for i = 1:length(files)
    [eeg,stim_events] = load_subject(files(i).name);
    all_eeg{i} = eeg;
    trial_order{i} = sort_trial_times(eeg,stim_events);
end

n_times = 7*64;
n_trials = 150;
n_chans = size(all_eeg{1}.trial{1},1);

x = zeros(n_trials*n_times,n_chans * length(all_eeg));

for i = 1:length(all_eeg)
    for trial_i = 1:length(trial_order{i})
        trial = trial_order{i}(trial_i);
        trial_time_indices = 1:min(size(all_eeg{i}.trial{trial},2),n_times);
        feature_indices = (i-1)*n_chans + ...
            (1:n_chans);
        time_indices = (trial_i-1)*n_times + (1:length(trial_time_indices));
        x(time_indices,feature_indices) = ...
            all_eeg{i}.trial{trial}(:,trial_time_indices)';
    end
end

% ======================================================================
% STEP 2: run mcca

chan_mean = mean(x,1);
x = x - chan_mean; % subtract mean from each column
C = x'*x; % covariance matrix

[A,score,AA] = nt_mcca(C,n_chans);

bar(score(1:100));
nkeep = 50; % number of components to keep

% Project out all but first "nkeep" components
for i = 1:length(all_eeg)
    iA = AA{i}; % subject-specific MCCA weights
    selection = zeros(size(iA,2),1);
    selection(1:nkeep) = 1;
    all_eeg{i}.old_trial = {};
    for t = 1:length(all_eeg{i}.trial)
        all_eeg{i}.old_trial{t} = all_eeg{i}.trial{t};
        arr = all_eeg{i}.trial{t};
        proj_arr = arr';
        mu = chan_mean((i-1)*n_chans + (1:n_chans));
        proj_arr = proj_arr - mu;
        proj_arr = proj_arr * (iA*diag(selection)*pinv(iA));
        all_eeg{i}.trial{t} = (proj_arr + mu)';
    end
end

% pre-cleaning plot configuration
plot_cfg = [];
plot_cfg.viewmode = 'vertical';
plot_cfg.preproc.detrend = 'yes';
plot_cfg.eegscale = 1;
plot_cfg.mychan = ft_channelselection('EX*',eeg);
plot_cfg.mychanscale = 1;
plot_cfg.ylim = [-20 20];

ft_databrowser(plot_cfg,all_eeg{3});
ft_databrowser(plot_cfg,cleaned_eeg{3});

% TODO: save these data and see if this "cleaned" result
% works any better (not super convinced it will, given that only 2
% components were found)

