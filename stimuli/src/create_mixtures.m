function create_mixtures(config)

    % GOOD GOD: I really wish people knew how to write *modular* functions.
    % Non-local relationships between variables are abundant here. So.
    % Confusing.  I would love to clean this code up more, but it is simply not
    % worth my time.
    %
    % The goals are:
    %  1. record all randomly generated values, so that we can regenerate
    %     the exact same stimulus set (this will make it possible to
    %     improve on the information I save, if I forget something)
    %  2. generate a highly comprehensible output with the information I
    %     currently want:
    %     a. the wave forms *after* being placed in space
    %     b. the target wave form before and after the insertion of the target
    %     c. the exact positions in space of each wave form (so we can
    %        reproduce the SOFA calls used to create this files).
    %     d. the names of each file mixed into a trial
    %
    % NOTE: no audio files will be saved, since these will
    % be generated as *.wav files.
    hrtfs = SOFAload(fullfile(basedir,'hrtfs','hrtf_b_nh172.sofa'));

    [audiodata,~] = read_sentences(config.sentence_dir,config.speaker_order);

    generate_stimuli(config,config.train_block_cfg,...
        fullfile(config.mix_dir,'training'),audiodata,hrtfs,1);
    generate_stimuli(config,config.test_block_cfg,...
        fullfile(config.mix_dir,'testing'),audiodat,hrtfs,0);
end

function d = isdir(d)
    if ~exist(d,'dir')
        mkdir(d)
    end
end

function generate_stimuli(config,block_cfg,indir,audiodata,hrtfs,is_training)
    isdir(indir);
    isdir(fullfile(indir,'target_component'))
    isdir(fullfile(indir,'mixture_components'))

    for trial_idx=1:size(trial_sentences,1)
        disp(trial_idx);

        % should the target be louder?
        if is_training
            mod_index = (mod(trial_idx-1,block_cfg.cond_rep)+1);
            repetitions = block_cfg.cond_rep;
            loud_target = mod_index <= repetitions/2;
        else
            loud_target = 0;
        end

        [stim,target,hrtf] = make_stim(config,block_cfg,trial_idx,audiodata,...
            hrtfs,loud_target);

        function saveto(str,stim)
            audiowrite(fullfile(indir,sprintf(str,trial_idx)),stim,config.fs);
        end

        saveto('trial_%02d.wav',stim);
        saveto(fullfile('target_component','trial_%02d.wav'),target);
        saveto(fullfile('mixture_components','trial_%02d_1.wav'),hrtf{1});
        saveto(fullfile('mixture_components','trial_%02d_2.wav'),hrtf{2});
        saveto(fullfile('mixture_components','trial_%02d_3.wav'),hrtf{3});
    end
end

function save_for_experiment(config,basedir)
%%
    block_cfg = config.train_block_cfg;
    bpath = 'stim_training/';
    save_this;

    block_cfg = config.test_block_cfg;
    bpath = '';
    save_this;

    function save_this
        trial_target_speakers = block_cfg.trial_target_speakers;
        [~,~,ac] = unique(trial_target_speakers,'stable');
        trial_target_speakers(trial_target_speakers>0) = ac(trial_target_speakers>0);

        trial_target_dir = block_cfg.trial_target_dir;
        [~,~,ac] = unique(trial_target_dir,'stable');
        trial_target_dir = ac;
        ctrl_idx = trial_target_dir(find(trial_target_speakers==-1,1));
        trial_target_dir(trial_target_dir==ctrl_idx) = trial_target_speakers(trial_target_dir==ctrl_idx);

        sal = trial_target_speakers>-1;
        this_info = [block_cfg.target_times sal trial_target_speakers trial_target_dir];
        dlmwrite([basedir bpath 'target_info_all.txt'],this_info,'delimiter',' ');

        sal = trial_target_speakers==1;
        this_info = [block_cfg.target_times sal trial_target_speakers trial_target_dir];
        dlmwrite([basedir bpath 'target_info_obj.txt'],this_info,'delimiter',' ');

        sal = trial_target_dir==1;
        this_info = [block_cfg.target_times sal trial_target_speakers trial_target_dir];
        dlmwrite([basedir bpath 'target_info_dir.txt'],this_info,'delimiter',' ');
    end

end

function [select_perms,select_perms_train] = ...
    select_trial_sentences(audiodata,num_trials,num_train)

    w = 0.3; % tolerance length difference
    sentence_perms = [];
    for s1=1:length(audiodata{1})
        for s2=1:length(audiodata{2})
            l1 = audiodata{1}.length_s(s1);
            l2 = audiodata{2}.length_s(s2);
            if abs(l1-l2)<w
                sentence_perms = [sentence_perms; s1 s2 abs(l1-l2)];
            end
        end
    end

    % find the most diverse set of params to select
    select_perms = [];
    [s1_perm_counts,s1_possibilities] = hist(sentence_perms(:,1),...
        unique(sentence_perms(:,1)));
    [s2_perm_counts,s2_possibilities] = hist(sentence_perms(:,2),...
        unique(sentence_perms(:,2)));
    s1_used_counts_p = zeros(1,length(s1_possibilities));
    s2_used_counts_p = zeros(1,length(s2_possibilities));
    while length(select_perms)<num_trials+num_train

        % first find which s1's were used least
        s1_least_used_p = find(s1_used_counts_p==min(s1_used_counts_p));
        % out of those, find the s1s with least perm pairs
        s1_least_used_perm_counts_p = s1_perm_counts(s1_least_used_p);
        % now we have to choose between these (indexed by possibilities,
        % not absolute indexing!
        s1_poss_p = s1_least_used_p(s1_least_used_perm_counts_p==...
            min(s1_least_used_perm_counts_p));
        % and here is the absolute indexing
        s1_poss = s1_possibilities(s1_poss_p);

        % now, for these s1, find every possible s2, absolute indexed
        s2_poss_all = sentence_perms(ismember(sentence_perms(:,1),s1_poss),1:2);
        % get the p indexed possibilities
        s2_poss_p = find(ismember(s2_possibilities,unique(s2_poss_all(:,2))));
        % eliminate the ones that weren't used 0 times
        s2_poss_p(s2_used_counts_p(s2_poss_p)~=min(s2_used_counts_p)) = [];
        if isempty(s2_poss_p)
            s1_used_counts_p(s1_poss) = s1_used_counts_p(s1_poss) + 1;
            continue;
        end
        % now get how many perms the s2's have
        s2_poss_perm_counts_p = s2_perm_counts(s2_poss_p);
        % then select the min one
        s2_least_used_perm_counts_p = find(s2_poss_perm_counts_p==...
            min(s2_poss_perm_counts_p));
        % if there is more than one, select one randomly
        asample = randsample(length(s2_least_used_perm_counts_p),1);
        least_used = s2_least_used_perm_counts_p(asample);
        s2_select = s2_possibilities(s2_poss_p(least_used));
        s2_select_p = find(s2_possibilities==s2_select);

        % now, find which s1's had this s2 as a pair
        s1_poss = s2_poss_all(s2_poss_all(:,2)==s2_select,1);
        % if there is more than one, select one randomly
        s1_select = s1_poss(randsample(length(s1_poss),1));
        s1_select_p = find(s1_possibilities==s1_select);

        % record the selected group
        select_perms = [select_perms; s1_select s2_select];

        % now update used counts
        s1_used_counts_p(s1_select_p) = s1_used_counts_p(s1_select_p) + 1;
        s2_used_counts_p(s2_select_p) = s2_used_counts_p(s2_select_p) + 1;

    end
    %%
    select_perms_train = select_perms(end-num_train+1:end,:);
    select_perms = select_perms(1:end-num_train,:);
    %%
    s3_lengths = audiodata{3}.length;
    s3_used_counts = zeros(1,length(s3_lengths));
    for select_idx=1:size(select_perms,1)
        ls = [audiodata{1}.length(select_perms(select_idx,1)) ...
            audiodata{2}.length(select_perms(select_idx,2))];
        s3_poss = find(abs(max(ls)-s3_lengths)<w);
        s3_poss_used_counts = s3_used_counts(s3_poss);
        s3_poss_least_used = s3_poss(s3_poss_used_counts==...
            min(s3_poss_used_counts));
        [~,s3_poss_least_used_closest_idx] = ...
            min(abs(max(ls)-s3_lengths(s3_poss_least_used)));
        s3_select = s3_poss_least_used(s3_poss_least_used_closest_idx);
        select_perms(select_idx,3) = s3_select;
        s3_used_counts(s3_select) = s3_used_counts(s3_select) + 1;
    end

    for select_idx=1:size(select_perms_train,1)
        ls = [audiodata{1}.length(select_perms(select_idx,1)) ...
            audiodata{2}.length(select_perms(select_idx,2))];
        [~,chosen3] = min(abs(max(ls)-s3_lengths));
        select_perms_train(select_idx,3) = chosen3;
    end

end

function [stim,target_stim,hrtf_stims] = make_stim(config,block_cfg,trial,...
    audiodata,hrtfs,loud_target)

    % get the targets
    idxs = block_cfg.trial_sentences(trial,:);
    s1 = audiodata{1}(idxs(1)).data;
    s2 = audiodata{2}(idxs(2)).data;
    s3 = audiodata{3}(idxs(3)).data;

    % insert the target sound
    [target_stim,sounds] = add_target(config,block_cfg,trial,{s1,s2,s3},..
        loud_target);
    [~,s1,s2,s3] = equalize_lengths(sounds{1},sounds{2},sounds{3});

    p1 = SOFAspat(s1,hrtfs,dir1,0);
    p2 = SOFAspat(s2,hrtfs,dir2,0);
    p3 = SOFAspat(s3,hrtfs,dir3,0);
    hrtf_stims = {p1,p2,p3}
    stim = p1 + p2 + p3;
end

function [target_sound,sounds] = add_target(config,block_cfg,trial,sounds,loud_target)
    target_t = block_cfg.target_indices(trial);
    target = block_cfg.target_speaker(trial);

    target_sound = manipulate(config,sounds{target},target_t);
    sounds{target} = target_sound

    function ss = manipulate(config,ss,target_t)
        %%
        idxs = target_t:target_t+floor(config.target_len*config.fs);
        if target_speaker==2
            seg = stretch(ss(idxs),config.analysis_len,config.synthesis_len)';
            seg = resample(seg,config.analysis_len,config.synthesis_len);
        else
            seg = stretch(ss(idxs),config.synthesis_len,config.analysis_len)';
            seg = resample(seg,config.synthesis_len,config.analysis_len);
        end
        seg = seg*rms(ss(idxs))/rms(seg);
        if loud_target
            seg = seg*10^(4/20);
        end
        %%
        ss = [ss(1:target_t); seg; ss(idxs(end)+1:end)];
        %%
        st = round(0.01*config.fs);
%             n = 5;
        idxs = target_t-st:target_t;
        ramp = linspace(0.7,1,length(idxs))';
        ss(idxs) = ss(idxs).*flip(ramp);
        idxs = target_t:target_t+st;
        ss(idxs) = ss(idxs).*ramp;
        idxs = (target_t-st:target_t)+config.fs*config.target_len;
        ss(idxs) = ss(idxs).*flip(ramp);
        idxs = (target_t:target_t+st)+config.fs*config.target_len;
        if length(ss)>idxs(end)
            ss(idxs) = ss(idxs).*ramp;
        end
    end
end
