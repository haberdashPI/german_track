run('../util/setup.m')
sid = 2

input_file = fullfile(data_dir,sprintf('eeg_response_%03d.bdf.mat',sid));
result_file = fullfile(data_dir,sprintf('eeg_response_%03d_ica.bdf.mat',sid));

dat = load(input_file)
data = dat.dat

% first, make sure there aren't any onerous artifacts
% (we're pretty conservative here, since the trials are fairly long)
cfg = [];
cfg.method = 'summary'
cfg.scalevertical = 20;
data = ft_rejectvisual(cfg,data);
% trial 84 removed

cfg = [];
cfg.viewmode = 'vertical'
ft_databrowser(cfg,data)

% note, you need to run eeglab to have binica in the path
cfg = []
cfg.method = 'binica'
comp = ft_componentanalysis(cfg, data);

alert()

% plot the components for visual inspection
figure
cfg = [];
cfg.component = 1:20;
cfg.layout = 'biosemi128.lay'
cfg.comment   = 'no';
ft_topoplotIC(cfg, comp)

figure
cfg = [];
cfg.component = 21:40;
cfg.layout = 'biosemi128.lay'
cfg.comment   = 'no';
ft_topoplotIC(cfg, comp)

figure
cfg = [];
cfg.component = 41:60;
cfg.layout = 'biosemi128.lay'
cfg.comment   = 'no';
ft_topoplotIC(cfg, comp)

% components 1, 41?
cfg = [];
cfg.layout = 'biosemi128.lay'
cfg.viewmode = 'component'
ft_databrowser(cfg,comp)

% just going with 1... 41 doesn't have a clear artifact
% signal.

cfg = [];
cfg.component = [1];
data_ica = ft_rejectcomponent(cfg,comp,data);

cfg = []
cfg.viewmode = 'vertical'
ft_databrowser(cfg,data);

cfg = []
cfg.viewmode = 'vertical'
ft_databrowser(cfg,data_ica);

ft_write_data(result_file,data_ica,'dataformat','matlab');