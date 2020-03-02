% setup paths
run(fullfile('..','..','src','matlab','util','setup.m'));
addpath(fullfile(base_dir,'src','matlab','salience'));
addpath(stimulus_dir);

% read in the location of the targets
config_file = fullfile(stim_data_dir,'config.json');
config = read_json(config_file);
target_locs = config.test_block_cfg.target_times(1:length(features));

% compute features for each stimulus
stimdir = fullfile(stim_data_dir,'mixtures','testing','target_component');
files = dir(fullfile(stimdir,'*.wav'));

features = cell(1,length(files));
file_lengths = zeros(length(features),1);
for fi = 1:length(files)
    filename = fullfile(stimdir,char(files(fi).name))
    [audio,fs] = audioread(filename);
    file_lengths(fi) = size(audio,1) / fs;
    if fs ~= 22050
        error("Expected a sample rate of 22050. Resample the audio first.");
    end
    features{1,fi} = Calc_Salience_Features(filename)';
end

% compute derivative-based salience
fs=1000/64; % caclculated based on line 48 of Cacl_Salience_Features.m
events_pos=Calc_Feat_Events(-1, features, fs);
events_neg=Calc_Feat_Events(1, features, fs);
events=[events_pos; events_neg];

% uncomment to use weighted salience
load(fullfile(base_dir,'src','matlab','salience','LDA_weights.mat'));
% uncomment to use unweighted salience
% LDA_W = 1;
salience=combined_salience(events, LDA_W, file_lengths);
maxsalience = max(cellfun(@max,salience));
minsalience = min(cellfun(@min,salience));
binstep = 0.5;
tiledlayout(6,7);
for k = 1:length(salience)
    nexttile;
    t = (0:(length(salience{k})-1)) * binstep;
    plot(t,salience{k});
    pos = [target_locs(k), minsalience, config.target_len, maxsalience - minsalience];
    rectangle('Position',pos,'FaceColor',[0.8 .1 .1 0.2],'EdgeColor','none');
    title(sprintf('Trial %02d',k))
    axis([0 max(t) minsalience maxsalience])
end

% compute salience of the targets
target_salience = zeros(length(salience),1);

for k = 1:length(salience)
    from = floor(target_locs(k)/binstep);
    to = ceil((target_locs(k)+config.target_len)/binstep);
    target_salience(k) = mean(salience{k}(from:to)) - mean_salience;
end
scatter(randn(length(target_salience),1),target_salience);

stimulus_index = (1:length(target_salience))';
salience = target_salience;
writetable(table(stimulus_index,salience),...
    fullfile(stim_data_dir,'target_salience.csv'));

