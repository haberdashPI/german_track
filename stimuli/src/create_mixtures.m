function create_mixtures(n_test_stimuli,n_train_stimuli,config)
    % create_mixtures   Generate the experimental stimuli by mixing sentences
    %
    % create_mixutres(n_test_stimuli,n_train_stimuli,config) - generate the given
    %   number of train and test stimuli.
    %
    %
    % The function draws on the sentences located in `stimuli/sentences`
    % and stores its results in `stimuli/mixtures`.
    %
    % Note that there are a number of installation specific
    % variables that have default values specific to my (David Little)
    % machine. You can change these using `config` or by just changing
    % their values within the function source.

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

    % ---------------------------------------------------------------------
    % running config ------------------------------------------------------
    % ---------------------------------------------------------------------

    mix_dir = fullfile(base_dir,'stimuli','mixtures');
    if ~exist(mix_dir,'dir')
        mkdir(mix_dir)
    end
    sentence_dir = fullfile(base_dir,'stimuli','sentences');

    % ---------------------------------------------------------------------
    % experiment/token setup ----------------------------------------------
    % ---------------------------------------------------------------------

    config = [];
    ppl_order = {'vadem','mensc','letzt'};
    [all_sentences,fs] = read_sentences(sentence_dir,ppl_order);

    hrtfs = SOFAload(fullfile(basedir,'hrtfs','hrtf_b_nh172.sofa'));
    config.sentence_fs = fs;
    config.all_sentences = all_sentences;

    config.normval = 5;
    config.analysis_len = 64;
    config.synthesis_len = 74;
    config.target_len = 1;
    config.switch_len = 1.2;
    config.min_stay_len = 0.5;
    config.jitter_period = 0.2;
    config.target_start_time = 1.5;

    test_block_cfg = [];
    test_block_cfg.target_cases = [1 1; 1 2; 2 1; 2 2; -1 -1];
    test_block_cfg.target_probs = [3;2;2;1;2]/10;
    test_block_cfg.num_trials = 50;

    train_block_cfg = [];
    train_block_cfg.target_cases = ...
        [1 1; -1 -1; 2 2; -1 -1; 1 2; 2 1; -1 -1; 1 1; 2 2; -1 -1];
    train_block_cfg.cond_rep = 4;
    train_block_cfg.target_probs = ...
        ones(size(train_block_cfg.target_cases,1),1)/...
            size(train_block_cfg.target_cases,1);
    train_block_cfg.num_trials = size(train_block_cfg.target_cases,1)*...
        train_block_cfg.cond_rep;

    % ---------------------------------------------------------------------
    % running of the actual sections for stims ----------------------------
    % ---------------------------------------------------------------------

    [select_perms,select_perms_train] = ...
      select_trial_sentences(config.all_sentences,...
        test_block_cfg.num_trials,train_block_cfg.num_trials);

    config.test_block_cfg = get_all_exp_stuff(config,test_block_cfg,...
        select_perms,0);
    config.train_block_cfg = get_all_exp_stuff(config,train_block_cfg,...
        select_perms_train,1);

    % TODO: don't save the audio files inside of the config file
    % these shoudl be stored as separate *.wav files.
    save(fullfile(base_dir,'stimuli','mixtures'),'config');
    save_for_experiment(config,basedir);
end

function block_cfg = configure_block(config,block_cfg,permutations)

    [trial_sentences,trial_target_speakers,trial_target_dir] = ...
        get_trial_info(permutations,config,block_cfg);

    block_cfg.trial_sentences = trial_sentences;
    block_cfg.trial_target_speakers = trial_target_speakers;
    block_cfg.trial_target_dir = trial_target_dir;

    block_cfg.target_times = zeros(size(trial_target_dir));
    block_cfg.switch_times = cell(size(trial_target_dir));
    block_cfg.directions = cell(size(trial_target_dir,1),3);

    for trial_idx=1:size(trial_sentences,1)
        block_cfg = make_directions(config,block_cfg,trial_idx);
        [tindex,ttime] = select_target_timing(config,block_cfg,trial_idx,
            % TODO: this is where I left off
            % TODO: figure out the remaining parameters
    end
end

function generate_stimuli(config,block_cfg,is_training)
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

        [stim,tt,h,d,critical_times] = ...
            make_stim(config,block_cfg,trial_idx,hrtfs,loud_target);

        % TODO: this is where I left off last time

        % TODO: the below lines need to be run earlier. They really all belong
        % in `configure_block` (in fact, this reveals the the `add_target`
        % function both figures out the exact location of the target and
        % modulates the sound, so I also need to separate those to things out
        % into separate functions

        block_cfg.target_times(trial_idx) = tt;
        block_cfg.switch_times{trial_idx} = critical_times;
        block_cfg.directions(trial_idx,:) = d;

        % TODO: save the mixutre, the sound with the target
        % and the unmixed HRTF modulated sources.
        save_trial(stim,h,config.fs,config.normval,basedir,thispath,...
            is_training);
    end
end

function [all_sentences,fs] = read_sentences(sentence_dir,ppl_order)
    reader_index = containers.Map;
    for i = 1:length(ppl_order)
        reader_index(ppl_order(i)) = i;
    end

    all_sentences = {[], [], []};
    all_sentence_files = sort(dir(fullfile(sentence_dir,'*.wav')));
    for file_idx=1:length(all_sentence_files)
        file_name = all_sentence_files(file_idx).name;
        [data,fs] = audioread(fullfile(sentence_dir,file_name));

        sentence = [];
        sentence.data = data;
        sentence.length_s = length(passage)/fs;
        sentence.filename = file_name;

        reader_id = file_name(1:5);
        if isKey(reader_index,reader_id)
            i = reader_index(reader_id);
            all_sentences{i} = ...
                [all_sentences{reader_index(reader_id)}; sentence];
        else
            error(['Could not find key for file prefix: ' reader_id])
        end
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

function save_trial(stim,handl,fs,normval,basedir,thispath,istraining)
%%
    stim = stim/normval;
    if ~istraining
        audio_path = [basedir 'stim_wav/' thispath '.wav'];
        img_path = [basedir 'stim_records/' thispath '.jpg'];
    else
        audio_path = [basedir 'stim_training/' thispath '.wav'];
        img_path = [basedir 'stim_training/' thispath '.jpg'];
    end
    audiowrite(audio_path,stim,fs);
    r = 150; % pixels per inch
    print(handl,'-dpng',sprintf('-r%d',r),img_path);
    close;
end

function [select_perms,select_perms_train] = ...
    select_trial_sentences(all_sentences,num_trials,num_train)

    w = 0.3; % tolerance length difference
    sentence_perms = [];
    for s1=1:length(all_sentences{1})
        for s2=1:length(all_sentences{2})
            l1 = all_sentences{1}.length_s(s1);
            l2 = all_sentences{2}.length_s(s2);
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
    s3_lengths = all_sentences{3}.length;
    s3_used_counts = zeros(1,length(s3_lengths));
    for select_idx=1:size(select_perms,1)
        ls = [all_sentences{1}.length(select_perms(select_idx,1)) ...
            all_sentences{2}.length(select_perms(select_idx,2))];
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
        ls = [all_sentences{1}.length(select_perms(select_idx,1)) ...
            all_sentences{2}.length(select_perms(select_idx,2))];
        [~,chosen3] = min(abs(max(ls)-s3_lengths));
        select_perms_train(select_idx,3) = chosen3;
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

function block_cfg = make_directions(config,block_cfg,trial)
    switch_len = config.switch_len;
    min_stay_len = config.min_stay_len;

    switch_num_range = 1:5;
    min_stim_len_for_switch = ...
        arrayfun(@(num_switch)((num_switch*2-1)*switch_len+...
            ceil(num_switch/2)*min_stay_len),switch_num_range);

    idxs = block_cfg.trial_sentences(trial,:);
    s1 = config.all_sentences{1}(idxs(1)).data;
    s2 = config.all_sentences{2}(idxs(2)).data;
    s3 = config.all_sentences{3}(idxs(3)).data;

    % compute directions (azimuth) of sounds
    len_stim = equalize_lengths(s1,s2,s3);
    [dir1,dir2,dir3] = ...
        make_directions(switch_num_range,min_stim_len_for_switch,len_stim);
    block_cfg.direction{trial,:} = {dir1,dir2,dir3};
end

function [stim,target_time,h,d,critical_times] = ...
    make_stim(config,block_cfg,trial,hrtfs,loud_target)

    % insert the target sound
    target = block_cfg.trial_target_speakers(trial);
    target_dir = block_cfg.trial_target_dir{trial};
    [s1,s2,s3] = add_target(config,target,target_dir,s1,s2,s3);

    [~,s1,s2,s3] = equalize_lengths(s1,s2,s3);

    p1 = SOFAspat(s1,hrtfs,dir1,0);
    p2 = SOFAspat(s2,hrtfs,dir2,0);
    p3 = SOFAspat(s3,hrtfs,dir3,0);
    stim = p1 + p2 + p3;
end

function [len_stim,s1,s2,s3] = equalize_lengths(s1,s2,s3)
    len_stim = max(length(s1),length(s2));
    s1 = [s1; zeros(len_stim-length(s1),1)];
    s2 = [s2; zeros(len_stim-length(s2),1)];
    s3 = [s3(1:min(len_stim,length(s3))); zeros(len_stim-length(s3),1)];
end

function [target_index,target_time] = ...
    select_target_timing(config,block_cfg,trial,target,directions)

    idxs = block_cfg.trial_sentences(trial,:);
    s1 = config.all_sentences{1}(idxs(1)).data;
    s2 = config.all_sentences{2}(idxs(2)).data;
    s3 = config.all_sentences{3}(idxs(3)).data;
    sounds = {s1,s2,s3};

    safety = 0.8;
    safety_end = config.fs*config.target_start_time;

    [ss,dd,dc,de,tt] = stream_select;
    [target_index,target_time] = tt_select(ss,dd,dc,de,tt)

    function [ss,dd,dc,de,tt] = stream_select
        ss = sounds{target}
        dd = directions{target}
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

function [s1,s2,s3,target_time] = add_target(config,target,target_dir,s1,s2,s3)
    if target>0
        manipulate;
        if target==1
            s1 = ss;
        elseif target==2
            s2 = ss;
        end
    end

    function manipulate
        %%
        idxs = target_t:target_t+config.target_len*config.fs;
        if target_speaker==2
            seg = E_phase(ss(idxs),config.analysis_len,config.synthesis_len)';
            seg = resample(seg,config.analysis_len,config.synthesis_len);
        else
            seg = E_phase(ss(idxs),config.synthesis_len,config.analysis_len)';
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
        idxs = (target_t-st:target_t)+config.fs*block_cfg.target_len;
        ss(idxs) = ss(idxs).*flip(ramp);
        idxs = (target_t:target_t+st)+config.fs*block_cfg.target_len;
        if length(ss)>idxs(end)
            ss(idxs) = ss(idxs).*ramp;
        end
    end
end


function [dir1,dir2,dir3] = ...
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
            extra_time - config.target_start_time,0,...
            extra_time.config.target_start_time);
        if target_dir=='right' || target_speaker==2
            section_lengths(1) = section_lengths(1) + config.target_start_time;
        elseif target_dir=='left'
            section_lengths(2) = section_lengths(2) + config.target_start_time;
        else
            section_lengths(1) = section_lengths(1) + config.target_start_time/2;
            section_lengths(2) = section_lengths(2) + config.target_start_time/2;
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


% function h = show_stim(config,sentence_idxs,azi1,azi_real1,s1,azi2,azi_real2,s2,azi3,azi_real3,s3,target_time,target_speaker,loud_target)

%     all_sentences = config.all_sentences;
%     fs = config.fs;
%     colors = ['b','r','g'];
%     rect_lightness = 0.8;
%     if target_speaker==1
%         target_color = ones(1,3)*rect_lightness+[0 0 1-rect_lightness];
%     elseif target_speaker==2
%         target_color = ones(1,3)*rect_lightness+[1-rect_lightness 0 0];
%     end
%     if target_speaker>0 && loud_target
%         target_color = target_color/1.5;
%     end

%     h = figure; hold on
%     yl = [-100 100];
%     if target_time>0
%         rectangle('Position',[target_time yl(1) config.target_len diff(yl)],'FaceColor',target_color,'linestyle','none');
%     end

%     azifactor = length(azi1)/length(s1);

%     t = (1:length(azi1))/fs/azifactor;

%     l1 = plot(t,azi1,'color',colors(1),'displayname',[num2str(all_sentences{1,2}) '-' num2str(sentence_idxs(1))]);
%     % keyboard
%     % plot(t,azi_real1','x','color',colors(1));
%     l2 = plot(t,azi2,'color',colors(2),'displayname',[num2str(all_sentences{2,2}) '-' num2str(sentence_idxs(2))]);
%     % plot(t,azi_real2','x','color',colors(2));
%     l3 = plot(t,azi3,'color',colors(3),'displayname',[num2str(all_sentences{3,2}) '-' num2str(sentence_idxs(3))]);
%     % plot(t,azi_real3','x','color',colors(3));

%     t = (1:length(s1))/fs;

%     plot(t,s1*40+45,'color',colors(1));
%     plot(t,s2*40-45,'color',colors(2));
%     plot(t,s3*40,'color',colors(3));

%     legend([l1 l2 l3],'location','northwest');
%     ylim(yl);
%     xlim([0 length(s1)/fs]);
%     set(gca,'ytick',[-90 90],'yticklabel',{'right','left'});
%     xlabel('Time (s)');

% end
