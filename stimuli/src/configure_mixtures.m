
function config = configure_mixtures(indir,config)
    mix_dir = 'mixtures';
    config.base_dir = indir;

    if ~exist(config.hrtf_file,'file')
        error(['Could not find specified HRTF file: ' hrtf_file])
    end


    sentence_dir = fullfile(indir,'sentences');
    if ~exist(sentence_dir,'dir')
        error(['The directory ' sentence_dir ' must exist and contain ' ...
               'all sentences used to create mixtures.']);
    end
    config.sentence_dir = sentence_dir;

    sentence_dir = 'sentences';
    [audiodata,fs] = read_sentences(fullfile(inddir,sentence_dir),...
        config.speaker_order);
    config.sentence_dir = sentence_dir;
    config.fs = fs;

    [select_perms,select_perms_train] = ...
      select_trial_sentences(audiodata,...
        config.test_block_cfg.num_trials,config.train_block_cfg.num_trials);

    config.test_block_cfg = configure_block(config,config.test_block_cfg,...
        select_perms,audiodata);
    config.train_block_cfg = configure_block(config,config.train_block_cfg,...
        select_perms_train,audiodata);

    keyboard
    save(fullfile(indir,mix_dir),'config');
end

function [select_perms,select_perms_train] = ...
    select_trial_sentences(audiodata,num_trials,num_train)

    tolerance_s = 0.3; % tolerance length difference
    sentence_perms = [];
    for s1=1:length(audiodata{1})
        for s2=1:length(audiodata{2})
            % TODO: this is the last place i ran into an error
            % MATLAB says: Expected one output from a curly brace or dot
            % indexing expression, but there were 68 results.
            l1 = audiodata{1}(s1).length_s;
            l2 = audiodata{2}(s2).length_s;
            if abs(l1-l2)<tolerance_s
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
    s3_lengths_s = [audiodata{3}.length_s];
    s3_used_counts = zeros(1,length(s3_lengths_s));
    for select_idx=1:size(select_perms,1)
        ls = [audiodata{1}(select_perms(select_idx,1)).length_s ...
            audiodata{2}(select_perms(select_idx,2)).length_s];
        s3_poss = find(abs(max(ls)-s3_lengths_s)<tolerance_s);
        s3_poss_used_counts = s3_used_counts(s3_poss);
        s3_poss_least_used = s3_poss(s3_poss_used_counts==...
            min(s3_poss_used_counts));
        [~,s3_poss_least_used_closest_idx] = ...
            min(abs(max(ls)-s3_lengths_s(s3_poss_least_used)));
        s3_select = s3_poss_least_used(s3_poss_least_used_closest_idx);
        select_perms(select_idx,3) = s3_select;
        s3_used_counts(s3_select) = s3_used_counts(s3_select) + 1;
    end

    for select_idx=1:size(select_perms_train,1)
        ls = [audiodata{1}(select_perms(select_idx,1)).length_s ...
            audiodata{2}(select_perms(select_idx,2)).length_s];
        [~,chosen3] = min(abs(max(ls)-s3_lengths_s));
        select_perms_train(select_idx,3) = chosen3;
    end

end


function block_cfg = configure_block(config,block_cfg,permutations,audiodata)

    [trial_sentences,trial_target_speakers,trial_target_dir] = ...
        get_trial_info(permutations,block_cfg);

    block_cfg.trial_sentences = trial_sentences;
    block_cfg.trial_target_speakers = trial_target_speakers;
    block_cfg.trial_target_dir = trial_target_dir;

    block_cfg.target_times = zeros(size(trial_target_dir));
    block_cfg.target_indices = zeros(size(trial_target_dir));
    block_cfg.switch_times = cell(size(trial_target_dir));
    block_cfg.s
    block_cfg.directions = cell(size(trial_target_dir,1),3);

    for trial_idx=1:size(trial_sentences,1)
        block_cfg = make_switches(config,block_cfg,trial_idx,audiodata);
        block_cfg = select_target_timing(config,block_cfg,trial_idx,audiodata);
    end
end

function [trial_sentences,trial_target_speakers,trial_target_dir] = ...
    get_trial_info(sentence_perms,block_cfg)

    target_cases = block_cfg.target_cases;
    target_probs = block_cfg.target_probs;
    num_trials = block_cfg.num_trials;

    trial_sentences = [];
    for i=1:size(target_cases,1)
        trial_sentences = [trial_sentences; ...
            repmat(target_cases(i,:),num_trials * target_probs(i),1)];
    end
    trial_sentences = [sentence_perms trial_sentences];
    trial_target_speakers = trial_sentences(:,end-1);
    trial_target_dir = categorical(trial_sentences(:,end),[-1 1 2],{'none','left','right'});
    trial_sentences = trial_sentences(:,1:3);
end

function block_cfg = make_switches(config,block_cfg,trial,audiodata)
    switch_len = config.switch_len;
    min_stay_len = config.min_stay_len;

    switch_num_range = 1:5;
    min_stim_len_for_switch = ...
        arrayfun(@(num_switch)((num_switch*2-1)*switch_len+...
                               ceil(num_switch/2)*min_stay_len),switch_num_range);

    idxs = block_cfg.trial_sentences(trial,:);
    l1 = length(audiodata{1}(idxs(1)).data):
    l2 = length(audiodata{2}(idxs(2)).data);
    l3 = length(audiodata{3}(idxs(3)).data);
    len_stim = equalize_lengths(l1,l2,l3);

    % jitter period and ampl
    A = 1/5;
    extra_times = len_stim/config.fs - min_stim_len_for_switch;

    % determine number of switches
    valid_switches = switch_num_range(extra_times>=0);
    switch_num = max(valid_switches);
    extra_time = extra_times(switch_num);

    % determine length of each section for s1
    if switch_num<3
        target_dir = block_cfg.trial_target_dir(trial);
        target_speaker = block_cfg.trial_target_speakers(trial);
        avail_time = extra_time - config.min_target_start;

        section_lengths = randfixedsum(switch_num+1,1,avail_time,0,avail_time);
        if target_dir=='right' || target_speaker==2
            section_lengths(1) = section_lengths(1) + config.min_target_start;
        elseif target_dir=='left'
            section_lengths(2) = section_lengths(2) + config.min_target_start;
        else
            section_lengths(1) = section_lengths(1) + config.min_target_start/2;
            section_lengths(2) = section_lengths(2) + config.min_target_start/2;
        end
    else
        section_lengths = randfixedsum(switch_num+1,1,extra_time,0,extra_time);
    end

    block_cfg.trial_section_lengths{trial} = section_lengths;
    block_cfg.trial_switch_num(trial) = switch_num;
end

function block_cfg = select_target_timing(config,block_cfg,trial,audiodata)
    idxs = block_cfg.trial_sentences(trial,:);
    s1 = audiodata{1}(idxs(1)).data;
    s2 = audiodata{2}(idxs(2)).data;
    s3 = audiodata{3}(idxs(3)).data;
    sounds = {s1,s2,s3};

    target = block_cfg.trial_target_speakers(trial);
    directions = make_directions(config,block_cfg,trial,audiodata);

    safety = 0.8;
    safety_end = config.fs*config.min_target_start;

    [ss,dd,dc,de,tt] = stream_select;
    [target_index,target_time] = tt_select(ss,dd,dc,de,tt);

    block_cfg.target_times(trial) = target_time;
    block_cfg.target_indices(trial) = target_index;

    function [ss,dd,dc,de,tt] = stream_select
        if target > 0
            ss = sounds{target};
            dd = directions{target};
            dc = circshift(dd,-round(config.target_len*config.fs));
            de = circshift(dd,round(safety*config.fs));
            tt = safety_end;
        else
            ss = []; dd = []; dc = []; de = []; tt = 0;
        end
    end

    function [target_index,target_time] = tt_select(ss,dd,dc,de,tt)
        target_dir = block_cfg.trial_target_dir(trial);
        if target_dir=='right'
            ffs = (dd(tt:end-safety_end)<-10).*(dc(tt:end-safety_end)<-10);
            get_optimal_t;
        elseif target_dir=='left'
            ffs = (dd(tt:end-safety_end)>37).*...
                (dc(tt:end-safety_end)>37).*...
                (de(tt:end-safety_end)>37);
            get_optimal_t;
        else
            target_index = 0;
        end
        target_time = target_index/config.fs;

        function get_optimal_t
            if isempty(find(ffs,1)), disp('FFS'); end
            sig_poss_start = abs(ss(tt+(1:length(ffs))).*ffs);
            [~,t_poss] = findpeaks(sig_poss_start,'npeaks',20,'sortstr',...
                'descend','minpeakdistance',config.fs/20);
            peak_rmss = zeros(length(t_poss),1);
            for ti=1:length(t_poss)
                peak_rmss(ti) = rms(ss(tt+t_poss(ti)+(1:config.fs)));
            end
            [~,t_selecti] = max(peak_rmss);
            target_index = t_poss(t_selecti)+tt;
        end
    end
end

