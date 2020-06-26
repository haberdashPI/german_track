run(fullfile('..','..','src','matlab','util','setup.m'));
mkdir(fullfile(cache_dir,'eeg'));

% whether to use previously preprocessed data stored in the cache
usecache = 1;

% if you are rerunning analyses you can set interactive to false; if you are
% analyzing a new subject, set this to true and run each section below,
% one at a time, using the plots to verify the results
interactive = false;

eegfiles = dir(fullfile(raw_datadir,'*.bdf'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot configuration

% trial plot configuration
plot_cfg = [];
plot_cfg.viewmode = 'vertical';
plot_cfg.eegscale = 1;
plot_cfg.ylim = [-20 20];
plot_cfg.preproc.detrend = 'no';
plot_cfg.preproc.demean = 'no';
plot_cfg.blocksize = 10;

% topographic alyout
elec = ft_read_sens(fullfile(raw_datadir,'biosemi64.txt'));
cfg = [];
cfg.elec = elec;
lay = ft_prepare_layout(cfg);
if interactive
    ft_layoutplot(cfg);
end

[closest,dists]=nt_proximity('biosemi64.lay',63);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% find the length of all stimuli

soundsdir = fullfile(stim_datadir,'mixtures','testing');
sounds = sort({dir(fullfile(soundsdir,'*.wav')).name});
sound_lengths = zeros(length(sounds),1);
for i = 1:length(sounds)
    [x,fs] = audioread(char(fullfile(soundsdir,sounds(i))));
    sound_lengths(i) = size(x,1) / fs;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% setup data cleaning parameters for individual participants

subject = [];

for i = 1:length(eegfiles)
    subject(i).load_channels = 1:70;
    subject(i).reref_first = false;
    subject(i).known_bad_channels = [28];
    subject(i).bad_channel_threshs = {3,150,2};
    subject(i).eye_pca_comps = 1;
    subject(i).eye_mask_threshold = 3;
    subject(i).segment_outlier_thresh = 3;
end

% Each subject should have a different entry below to tune the data cleaning
% parameters to their data.

subject(1).sid = 8;

subject(2).sid = 9;
subject(2).load_channels = [1:64,129:134];

subject(3).sid = 10;
subject(3).known_bad_channels = [28,57];
subject(3).eye_pca_comps = 2;

subject(4).sid = 11;
subject(4).eye_pca_comps = 1;

subject(5).sid = 12;
subject(5).eye_pca_comps = 3;

subject(6).sid = 13;

subject(7).sid = 14;

% subject 15 has no good data, file not generated

subject(8).sid = 16;
subject(8).known_bad_channels = [4,28];

subject(9).sid = 17;
subject(9).reref_first = true;

subject(10).sid = 18;

subject(11).sid = 19;
subject(11).known_bad_channels = [28,57];

subject(12).sid = 20;
subject(12).sid = [];

subject(13).sid = 21;

subject(14).sid = 22;
subject(14).reref_first = true;

subject(15).sid = 23;
subject(15).sid = [];

subject(16).sid = 24;

subject(17).sid = 25;

subject(18).sid = 26;
subject(18).sid = [];

subject(19).sid = 27;
subject(19).known_bad_channels = [22,28];

subject(20).sid = 28;
subject(20).known_bad_channels = [28,57];

subject(21).sid = 29;

subject(22).sid = 30;

subject(23).sid = 31;
subject(23).known_bad_channels = [16,24,28,57,60,61];

subject(24).sid = 32;
subject(24).known_bad_channels = [28,63];

subject(25).sid = 33;
subject(25).known_bad_channels = [5,28];

subject(26).sid = 34;
subject(26).reref_first = true;
subject(26).bad_channel_threshs = {2,150,1};

subject(27).sid = 35;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% data cleaning

data = [];

for i = 1:length(eegfiles)

    %% file information
    filename = eegfiles(i).name;
    filepath = fullfile(raw_datadir,filename);
    sid = gt_sidforfile(filename);
    if subject(i).sid ~= sid
        error('Wrong subject id (%d) specified for index %d, expected sid %d',...
            subject(i).sid,i,sid);
    end

    savename = regexprep(filename,'.bdf$','.eeg');
    savetopath = fullfile(cache_dir,'eeg',savename);

    if isempty(subject(i).sid)
        warning("Subject id %d will be ignored.",sid);
        continue
    end
    if isfile(savetopath) && usecache
        warning("Using cached subject data for sid %d.",sid)
        continue
    end

    %% read in the events
    event_file = fullfile(processed_datadir,'eeg',sprintf('sound_events_%03d.csv',sid));
    stim_events = readtable(event_file);

    %% read in eeg data header
    eeg = gt_loadbdf(filepath,stim_events,'lengths',sound_lengths,'channels',...
        subject(i).load_channels);
    if subject(i).reref_first
        eegcat = gt_fortrials(@(x)x,eeg);
        eegcat = vertcat(eegcat{:});
        eeg = gt_asfieldtrip(eeg,nt_rereference(eegcat));
    end

    %% preprocess data
    eeg = gt_downsample(eeg,stim_events,8);
    eeg = gt_settrials(@nt_demean,eeg);
    if interactive
        ft_databrowser(plot_cfg, eeg);
    end

    eeg = gt_settrials(@gt_interpolate_bad_channels,eeg,...
        subject(i).known_bad_channels,closest,dists,'channels',1:64);
    [eeg,w] = gt_settrials(@gt_detrend,eeg,[1 10],'progress','detrending...');

    %% find bad channels
    freq = 0.5;
    bad_indices = gt_fortrials(@nt_find_bad_channels,eeg,freq,...
        subject(i).bad_channel_threshs{:},'channels',1:64);
    if interactive
        % run below line to see which indices are bad for a given trial
        eeg.hdr.label(bad_indices{1})
        this_plot = plot_cfg;
        this_plot.preproc.detrend = 'yes';
        ft_databrowser(plot_cfg, eeg);
    end

    %% interpolate bad channels
    eeg = gt_settrials(@gt_interpolate_bad_channels,{eeg,bad_indices},...
        closest,dists,'channels',1:64);
    if interactive
        ft_databrowser(plot_cfg, eeg);
    end

    % eliminate eyblinks
    [weye,pcas] = gt_mask_eyeblinks(eeg,w,67:70,subject(i).eye_pca_comps,...
        subject(i).eye_mask_threshold);

    if interactive

        chans = cellfun(@(x)sprintf('pc%02d',x),num2cell(1:4),...
            'UniformOutput',false);
        this_plot = plot_cfg;
        this_plot.ylim = [-400 400];
        ft_databrowser(this_plot, gt_asfieldtrip(eeg,pcas,'label',chans,...
            'cropfirst',10));

        eyemask = min(horzcat(weye{:}))';
        eegcat = gt_fortrials(@(x)x,eeg);
        eegcat = vertcat(eegcat{:});
        chans = [ eeg.label; 'mask' ];
        ft_databrowser(plot_cfg, gt_asfieldtrip(eeg,[eegcat 100.*eyemask],...
            'label',chans))

    end

    % eegcat = gt_fortrials(@(x)x,eeg);
    % eegcat = vertcat(eegcat{:});
    % weyecat = horzcat(weye{:})';
    % [toeye,p1,p2] = nt_dss0(nt_cov(eegcat),nt_cov(bsxfun(@times,eegcat,1-weyecat)));
    % eegcat = nt_tsr(eegcat,eegcat*toeye(:,comps));
    % eegeye = gt_asfieldtrip(eeg,eegcat);
    % ft_databrowser(plot_cfg,eegeye);

    % if interactive

    %     plot(p2./p1, '.-');

    %     comps = 1:4

    %     % plot timecourse of components
    %     chans = cellfun(@(x)sprintf('eye%02d',x),num2cell(comps),'UniformOutput',false);
    %     this_plot = plot_cfg;
    %     this_plot.ylim = [-1 1] .* 1e-1;
    %     ft_databrowser(this_plot, gt_asfieldtrip(eeg,eegcat*toeye(:,comps),...
    %         'label',chans,'croplast',10))

    %     % plot components
    %     topo = [];
    %     topo.component = 1:length(comps);
    %     topo.layout = lay;
    %     figure; ft_topoplotIC(topo,gt_ascomponent(eeg,toeye(:,comps)));

    % end

    wcomb = cellfun(@(x,y) min(x,y'), w, weye', 'UniformOutput', false);
    [eegoutl,woutl] = gt_settrials(@gt_outliers,{eeg,wcomb},5,3,false,...
        'progress','finding outliers...'); % like nt_outliers, but shows a progress bar
    if interactive
        ft_databrowser(plot_cfg, eegoutl);
        eegw = eegoutl
        eegw.trial = cellfun(@(x) x',woutl','UniformOutput',false);
        this_plot = plot_cfg;
        this_plot.ylim = [0 1];
        ft_databrowser(this_plot, eegw);
    end

    %% weighted rerefence
    eegcat = gt_fortrials(@(x)x,eeg);
    eegcat = vertcat(eegcat{:});
    eeg = gt_asfieldtrip(eeg,nt_rereference(eegcat));
    if interactive
        ft_databrowser(plot_cfg, eeg);
    end

    %% detect outlying segments
    wcomb = cellfun(@(x,y) min(x',y), woutl', weye, 'UniformOutput', false);
    [wseg,segnorm,segsd] = ...
        gt_segment_outliers(eeg,wcomb,subject(i).segment_outlier_thresh,0.25);

    if interactive

        imagesc(segsd);

        plot(segnorm,'.-')
        mean(segnorm > subject(i).segment_outlier_thresh)

        segmask = min(horzcat(wseg{:}))';
        eegcat = gt_fortrials(@(x)x,eeg);
        eegcat = vertcat(eegcat{:});
        chans = [ eeg.label; 'mask' ];
        ft_databrowser(plot_cfg, gt_asfieldtrip(eeg,[eegcat 20.*segmask],...
            'label',chans));

    end

    for j = 1:length(wseg)
        if size(wseg{j},1) ~= 70
            error("Unexpected number of channels")
        end
        wseg{j}(bad_indices{j},:) = 0;
        wseg{j}(subject(i).known_bad_channels,:) = 0;
    end

    save_subject_binary(eeg,savetopath,'weights',wseg);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute correlation (C) of channel-subject features across all
% stimulus-condition pairs

cleaned_files = dir(fullfile(cache_dir,'eeg','*.eeg'));
maxlen = round(256*(max(sound_lengths)+0.5));

cachefile = fullfile(cache_dir,'eeg','C.mat')
if usecache && isfile(cachefile)
    load(fullfile(cache_dir,'eeg','C.mat'))
else
    C = gt_mcca_C(cleaned_files,maxlen,{'global','object','spatial'},1:50,1:64);
    save(cachefile,'C');
end

[A,score,AA] = nt_mcca(C,64);

if interactive
    imagesc(log(abs(C)));
    imagesc(C);

    plot(score(1:60),'.-');
    plot(score,'.-');

    total = 0;
    for i = 1:length(cleaned_files)
        total = total + all(all(AA{i} == 0));
    end
    % total should be 0

    chweights = zeros(1:64,length(cleaned_files))
    for i = 1:length(cleaned_files)
        filename = cleaned_files(i).name;
        filepath = fullfile(cache_dir,'eeg',filename);
        sid = gt_sidforfile(filename);
        [~,~,w] = load_subject_binary(filepath);

        wcat = horzcat(w{:});
        chweights(:,i) = mean(wcat(1:64,:),2);
    end

    % plot components of all particiapnts
    figure; tiledlayout(5,5);
    for i = 1:length(cleaned_files)
        nexttile;

        sid = gt_sidforfile(cleaned_files(i).name);

        im = AA{i}(:,1:nkeep);
        im = im.*chweights(:,i);
        imagesc(im);
        title(sprintf("Subject %d",sid));
    end

    % examine MCCA cleaned data of selected individual participants
    i = 24; % select participant here
    nkeep = 30;

    filename = cleaned_files(i).name
    filepath = fullfile(cache_dir,'eeg',filename);

    [trial,label,w] = load_subject_binary(filepath);
    raw = gt_eeg_to_ft(trial,label,256);
    mcca = project_mcca(raw,w,nkeep,1:64,AA{i},0);

    this_plot = plot_cfg;
    this_plot.channel = mcca.label;
    ft_databrowser(this_plot,raw);
    this_plot.ylim = [-1 1].*1e0;
    ft_databrowser(this_plot,mcca);

    mcca_comp = mcca;
    mcca_comp.trial = mcca_comp.projected;
    mcca_comp.label = cellfun(@(x)sprintf('comp%02d',x),num2cell(1:nkeep),...
        'UniformOutput',false)
    this_plot.channel = mcca_comp.label;
    this_plot.ylim = [-1 1].*1e-1;
    ft_databrowser(this_plot,mcca_comp);

    comps = 1:4
    topo = [];
    topo.component = 1:length(comps);
    topo.layout = lay;
    figure; ft_topoplotIC(topo,gt_ascomponent(mcca,AA{i}(:,comps).*chweights(:,i)));

end

% save the subjects using the best MCCA components
nkeep = 30;
for i = 1:length(cleaned_files)
    filename = cleaned_files(i).name;
    filepath = fullfile(cache_dir,'eeg',filename);
    sid = gt_sidforfile(filepath);

    [trial,label,w] = load_subject_binary(filepath);
    raw = gt_eeg_to_ft(trial,label,256);
    mcca = project_mcca(raw,w,nkeep,1:64,AA{i},0);

    savename = regexprep(cleaned_files(i).name,'.eeg$','.mcca');
    mccafile = fullfile(processed_datadir,'eeg',savename);

    save_subject_components(mcca,mccafile)
end
