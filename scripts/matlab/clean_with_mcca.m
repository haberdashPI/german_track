run(fullfile('..','..','src','matlab','util','setup.m'));

% ======================================================================
% STEP 0: configuration

[eeg,~,~] = load_subject('eeg_response_008.mat');

plot_cfg = [];
plot_cfg.viewmode = 'vertical';
plot_cfg.preproc.detrend = 'yes';
plot_cfg.eegscale = 1;
plot_cfg.mychan = ft_channelselection('EX*',eeg);
plot_cfg.mychanscale = 1;
plot_cfg.ylim = [-1 1];

databrowse_cfg          = [];
databrowse_cfg.method   = 'summary';
databrowse_cfg.alim     = 1e-12;

% ======================================================================
% STEP 1: load cleaned data
files = dir(fullfile(data_dir,'*.mat'));
files = files(cellfun(@(x) ~isempty(regexp(x,'[0-9]{3}_cleaned\.mat')),{files.name}));

all_eeg = {};
trial_order = {};
for i = 1:length(files)
    [eeg,stim_events,sid] = load_subject(files(i).name);
    all_eeg{i} = eeg;
    trial_order{i} = sort_trial_times(eeg,stim_events);
    all_eeg{i}.raw_trial = all_eeg{i}.trial;
    all_eeg{i}.sid = sid;
end

% save files to easily read binary files
% these can be easily loaded in other languages
global data_dir
for i = 1:length(files)
    name = files(i).name;
    name = regexprep(name,'.mat$','.eeg');
    save_subject_binary(all_eeg{i},fullfile(data_dir,name))
end

n_times = 7*64;
n_trials = 150;
n_chans = size(all_eeg{1}.trial{1},1);

x = zeros(n_trials*n_times,n_chans * length(all_eeg));

for i = 1:length(all_eeg)
    for trial_i = 1:length(trial_order{i})
        trial = trial_order{i}(trial_i);
        trial_time_indices = 1:min(size(all_eeg{i}.trial{trial},2),n_times);
        feature_indices = (i-1)*n_chans + (1:n_chans);
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

bar(score(1:300));
nkeep = 9;

% Project out all but first "nkeep" components
for i = 1:length(all_eeg)
    mu = chan_mean((i-1)*n_chans + (1:n_chans));
    all_eeg{i} = project_mcca(all_eeg{i},nkeep,AA{i},mu);
end

ft_databrowser(databrowse_cfg,all_eeg{9});

% ft_databrowser(plot_cfg,all_eeg{3});
% ft_databrowser(plot_cfg,cleaned_eeg{3});

for i = 1:length(all_eeg)
    filename = sprintf('eeg_response_%03d_mcca%02d.mcca_proj',all_eeg{i}.sid,nkeep);
    save_subject_components(all_eeg{i}, fullfile(data_dir,filename));
end
