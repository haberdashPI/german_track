
function config = configure_mixtures(indir,config)
    mix_dir = isdir(fullfile(indir,'mixtures'));
    config.mix_dir = mix_dir;

    if ~exist(config.hrtf_file,'file')
        error(['Could not find specified HRTF file: ' hrtf_file])
    end


    sentence_dir = fullfile(indir,'sentences');
    if ~exist(sentence_dir,'dir')
        error(['The directory ' sentence_dir ' must exist and contain ' ...
               'all sentences used to create mixtures.']);
    end
    config.sentence_dir = sentence_dir;

    sentence_dir = fullfile(indir,'sentences');
    [audiodata,fs] = read_sentences(sentence_dir,config.speaker_order);
    config.sentence_dir = sentence_dir;
    config.fs = fs;

    [select_perms,select_perms_train] = ...
      select_trial_sentences(audiodata,...
        config.test_block_cfg.num_trials,config.train_block_cfg.num_trials);

    config.test_block_cfg = configure_block(config,config.test_block_cfg,...
        select_perms,audiodata);
    config.train_block_cfg = configure_block(config,config.train_block_cfg,...
        select_perms_train,audiodata);

    save(mix_dir,'config');
end

function [select_perms,select_perms_train] = ...
    select_trial_sentences(audiodata,num_trials,num_train)

    w = 0.3; % tolerance length difference
    sentence_perms = [];
    for s1=1:length(audiodata{1})
        for s2=1:length(audiodata{2})
            % TODO: this is the last place i ran into an error
            % MATLAB says: Expected one output from a curly brace or dot
            % indexing expression, but there were 68 results.
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


function block_cfg = configure_block(config,block_cfg,permutations,audiodata)

    [trial_sentences,trial_target_speakers,trial_target_dir] = ...
        get_trial_info(permutations,config,block_cfg);

    block_cfg.trial_sentences = trial_sentences;
    block_cfg.trial_target_speakers = trial_target_speakers;
    block_cfg.trial_target_dir = trial_target_dir;

    block_cfg.target_times = zeros(size(trial_target_dir));
    block_cfg.target_indices = zeros(size(trial_target_dir));
    block_cfg.switch_times = cell(size(trial_target_dir));
    block_cfg.directions = cell(size(trial_target_dir,1),3);

    for trial_idx=1:size(trial_sentences,1)
        block_cfg = make_directions(config,block_cfg,trial_idx,audiodata);
        block_cfg = select_target_timing(config,block_cfg,trial_idx,audiodata)
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

function [dir1,dir2,dir3,critical_times] = ...
    make_directions(config,switch_num_range,min_stim_len_for_switch,len_stim)

    % jitter period and ampl
    A = 1/5;
    extra_time = len_stim/config.fs - min_stim_len_for_switch;

    % determine number of switches
    valid_switches = switch_num_range(extra_time>=0);
    switch_num = max(valid_switches);

    % determine length of each section for s1
    if switch_num<3
        section_lengths = randfixedsum(switch_num+1,1,...
            extra_time - config.min_target_start,0,...
            extra_time.config.min_target_start);
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

    % put the opening section
    switch_wave = (0:1/(fs*switch_len):1)';

    % then put everything together
    critical_times = [];
    sec1;
    sec2;
    sec3;
    secs_equalize;
    critical_times = cumsum(critical_times);
    to_angle;

    function sec1
        dir1 = make_jitter(round((section_lengths(1)+min_stay_len)*fs),0);
        dir2 = make_jitter(length(dir1),1);
        sl = min(length(dir2),round(length(switch_wave)/2));
        dir3 = flip([switch_wave(1:sl); make_jitter(length(dir1)-sl,1)-switch_wave(sl)]);
        critical_times = [critical_times round((section_lengths(1)+min_stay_len)*fs)];
    end

    function sec2
        for section_idx=2:switch_num
            if dir1(end)<0.5
                this_switch = switch_wave;
                this_len = round((section_lengths(section_idx)+switch_len*2)*fs);
                this_len2 = this_len-length(switch_wave)*2;
                dir2 = [dir2; make_jitter(length(this_switch),1,1); flip(switch_wave); make_jitter(this_len2,0,1); switch_wave];
                critical_times = [critical_times length(this_switch) length(switch_wave) this_len2 length(switch_wave)];
                mini_switch = switch_wave(1:ceil(this_len2/2));
                dir3 = [dir3; make_jitter(length(this_switch)*2,0); mini_switch; flip(mini_switch(1:end-mod(this_len2,2))); make_jitter(length(switch_wave),0)];
            else
                this_switch = flip(switch_wave);
                this_len = round((section_lengths(section_idx)+min_stay_len)*fs);
                dir2 = [dir2; make_jitter(length(switch_wave)+this_len,1)];
                mini_switch = switch_wave(1:ceil(this_len/2));
                dir3 = [dir3; make_jitter(length(switch_wave),0); mini_switch; flip(mini_switch(1:end-mod(this_len,2)))];
                critical_times = [critical_times length(this_switch) this_len];
            end
            dir1 = [dir1; this_switch; make_jitter(this_len,this_switch(end))];
        end
    end

    function sec3
        section_idx = switch_num + 1;
        if dir1(end)<0.5
            this_switch = switch_wave;
            this_len = round((section_lengths(section_idx))*fs);
            dir2 = [dir2; make_jitter(length(this_switch),1); flip(switch_wave(end+1-min(this_len,length(switch_wave)):end)); make_jitter(this_len-length(switch_wave),0)];
            critical_times = [critical_times length(this_switch) length(switch_wave)];
        else
            this_switch = flip(switch_wave);
            this_len = round((section_lengths(section_idx))*fs);
            dir2 = [dir2; make_jitter(length(switch_wave)+this_len,1)];
            critical_times = [critical_times length(this_switch) this_len];
        end
        dir1 = [dir1; this_switch; make_jitter(this_len,this_switch(end))];
        dir3 = [dir3; make_jitter(length(this_switch)+this_len,0)];
    end

    function secs_equalize
        if ~isempty(find(diff([length(dir1) length(dir2) length(dir3)])~=0,1))
            disp('dir lengths not the same!');
        end
        dir1 = dir1(1:len_stim);
        dir2 = dir2(1:len_stim);
        dir3 = dir3(1:len_stim);
    end

    function jit = make_jitter(len_requested,direc,override)
        if nargin==2, override=0; end
        jit = -sin(2*pi/(config.jitter_period)*...
            linspace(0,len_requested/fs,len_requested))'*A;
        if direc==1 && override
            jit = jit+direc;
        elseif direc==1 || override
            jit = -jit+direc;
        end
    end

    function to_angle
        if d1s0
            dir1 = (dir1+A)/(1+A*2)*180-90;
            dir2 = (dir2+A)/(1+A*2)*180-90;
            dir3 = (dir3+A)/(1+A*2)*180-90;
        else
            dir1 = 90-(dir1+A)/(1+A*2)*180;
            dir2 = 90-(dir2+A)/(1+A*2)*180;
            dir3 = 90-(dir3+A)/(1+A*2)*180;
        end
    end
end

function [target_index,target_time] = select_target_timing(config,...
    block_cfg,trial,audiodata)

    idxs = block_cfg.trial_sentences(trial,:);
    s1 = audiodata{1}(idxs(1)).data;
    s2 = audiodata{2}(idxs(2)).data;
    s3 = audiodata{3}(idxs(3)).data;
    sounds = {s1,s2,s3};

    target = block_cfg.trial_target_speakers(trial);
    directions = block_cfg.direction{trial,:};

    safety = 0.8;
    safety_end = config.fs*config.min_target_start;

    [ss,dd,dc,de,tt] = stream_select;
    [target_index,target_time] = tt_select(ss,dd,dc,de,tt);

    block_cfg.target_times(trial_idx) = target_time;
    block_cfg.target_indices(trial_idx) = target_index;

    function [ss,dd,dc,de,tt] = stream_select
        ss = sounds{target};
        dd = directions{target};
        dc = circshift(dd,-round(config.target_len*fs));
        de = circshift(dd,round(safety*fs));
        tt = safety_end;
    end

    function [target_index,target_time] = tt_select(ss,dd,dc,de,tt)
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
        target_time = target_index/fs;

        function get_optimal_t
            if isempty(find(ffs,1)), disp('FFS'); end
            sig_poss_start = abs(ss(tt+(1:length(ffs))).*ffs);
            [~,t_poss] = findpeaks(sig_poss_start,'npeaks',20,'sortstr','descend','minpeakdistance',fs/20);
            peak_rmss = zeros(length(t_poss),1);
            for ti=1:length(t_poss)
                peak_rmss(ti) = rms(ss(tt+t_poss(ti)+(1:fs)));
            end
            [~,t_selecti] = max(peak_rmss);
            target_index = t_poss(t_selecti)+tt;
        end
    end
end

