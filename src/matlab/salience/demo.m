%% Extract a set of acoustic features

feats=nick_features('FilewithPaths.txt', n_files);

%% Find events using derivative analysis
fs=1000/64;
events_pos=Calc_Feat_Events(-1, feats, fs);
events_neg=Calc_Feat_Events(1, feats, fs);
events=[events_pos; events_neg];

%% combine with LDA weights
load('LDA_weights.mat')
salience=combined_salience(events, LDA_W, fs);


