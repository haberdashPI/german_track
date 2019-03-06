function E_stimuli(run_main,run_train)

    % ---------------------------------------------------------------------
    % running config ------------------------------------------------------
    % ---------------------------------------------------------------------
    
    if nargin==0
        run_main = 1;
        run_train = 0;
        runexample = 0;
    else
        runexample = 0;
    end
%     run('/Users/merve/Dropbox/Grad/External tools/sofa-api-mo-1.0.1/API_MO/SOFAstart.m')
    
    % ---------------------------------------------------------------------
    % experiment/token setup ----------------------------------------------
    % ---------------------------------------------------------------------

    experiment_cfg = [];
    if 1
        folder_name = 'passages1 levelatord';
        [all_sentences,fs] = read_sentences(folder_name);
    end
    path = 'Experiments/E1/';
    experiment_cfg.fs = fs;
    experiment_cfg.all_sentences = all_sentences;
    
    experiment_cfg.normval = 5;
    experiment_cfg.analysis_len = 64;
    experiment_cfg.synthesis_len = 72;
    experiment_cfg.dev_len = 0.8; 
    
    test_block_cfg = [];
    test_block_cfg.dev_cases = [1 1; 2 1; 1 2; 2 2; -1 -1];
    test_block_cfg.dev_probs = [2;1;3;2;2]/10;
    test_block_cfg.num_trials = 10;  
    
    train_block_cfg = [];
    train_block_cfg.dev_cases = [1 1; -1 -1; 2 2; -1 -1; 1 2; 2 1; -1 -1; 1 1; 2 2; -1 -1];
    train_block_cfg.dev_probs = ones(size(train_block_cfg.dev_cases,1),1)/size(train_block_cfg.dev_cases,1);
    train_block_cfg.num_trials = size(train_block_cfg.dev_cases,1);  
    
    % ---------------------------------------------------------------------
    % running of the actual sections for stims ----------------------------
    % ---------------------------------------------------------------------
    
    [select_perms,select_perms_train] = get_sentences_per_trial(experiment_cfg.all_sentences,test_block_cfg.num_trials,train_block_cfg.num_trials);
    
    if run_main
        experiment_cfg.block_cfg = test_block_cfg;
        sentence_perms = select_perms;
        get_all_exp_stuff(0);
        experiment_cfg.block_cfg.trial_sentences = trial_sentences;
        experiment_cfg.block_cfg.trial_dev_speakers = trial_dev_speakers;
        experiment_cfg.block_cfg.trial_dev_direction = trial_dev_direction;
        experiment_cfg.block_cfg.target_times = target_times;
        experiment_cfg.test_block_cfg = experiment_cfg.block_cfg;
    end
    
    if run_train
        experiment_cfg.block_cfg = train_block_cfg;
        sentence_perms = select_perms_train;
        get_all_exp_stuff(1);
        experiment_cfg.block_cfg.trial_sentences = trial_sentences;
        experiment_cfg.block_cfg.trial_dev_speakers = trial_dev_speakers;
        experiment_cfg.block_cfg.trial_dev_direction = trial_dev_direction;
        experiment_cfg.block_cfg.target_times = target_times;
        experiment_cfg.train_block_cfg = experiment_cfg.block_cfg;
    end
    
    if runexample
        experiment_cfg = make_all_stim(experiment_cfg,path);
    end
    
    if run_main==1 && run_train==1
        experiment_cfg = rmfield(experiment_cfg,'block_cfg');
        save([path 'experiment_record.mat']);
        save_for_experiment(experiment_cfg,path);
    end
    
    % ---------------------------------------------------------------------
    % nest functions ------------------------------------------------------
    % ---------------------------------------------------------------------
    
    function get_all_exp_stuff(is_training)
        [trial_sentences,trial_dev_speakers,trial_dev_direction] = get_trial_info(sentence_perms,experiment_cfg);
        target_times = zeros(size(trial_dev_direction));
        for trial_idx=1:size(trial_sentences,1)
%         for trial_idx=1:10
            thispath = sprintf('trial_%d',trial_idx); 
            [stim,tt,h] = make_stim(experiment_cfg,trial_sentences(trial_idx,:),trial_dev_speakers(trial_idx),trial_dev_direction(trial_idx));
            target_times(trial_idx) = tt;
            save_trial(stim,h,experiment_cfg.fs,experiment_cfg.normval,path,thispath,is_training);
        end
    end
    
end

function [all_sentences,fs] = read_sentences(folder_name) 
%%
    all_sentences = cell(0,2);
    all_sentence_files = dir(['Experiments/E1/' folder_name '/']);
    for file_idx=1:length(all_sentence_files)
        file_name = all_sentence_files(file_idx).name;
        if length(file_name)<3 || ~strcmp(file_name(end-2:end),'wav')
            continue;
        end
        [passage,fs] = audioread(['Experiments/E1/' folder_name '/' file_name]);
        reader_id = file_name(1:5);
        all_sentences_idx = find(ismember(all_sentences(:,2),reader_id));
        if isempty(all_sentences_idx)
            all_sentences = [all_sentences; {[{passage(:,1)} {length(passage)/fs} {file_name}] reader_id}];
        else
            all_sentences{all_sentences_idx} = [all_sentences{all_sentences_idx}; {passage(:,1)} {length(passage)/fs} {file_name}];
        end
    end
    
end

function save_for_experiment(experiment_cfg,path)
%%
    block_cfg = experiment_cfg.train_block_cfg;
    bpath = 'stim_training/';
    save_this;
    
    block_cfg = experiment_cfg.test_block_cfg;
    bpath = '';
    save_this;    
    
    function save_this
        trial_dev_speakers = block_cfg.trial_dev_speakers;
        [~,~,ac] = unique(trial_dev_speakers,'stable');
        trial_dev_speakers(trial_dev_speakers>0) = ac(trial_dev_speakers>0);

        trial_dev_direction = block_cfg.trial_dev_direction;
        [~,~,ac] = unique(trial_dev_direction,'stable');
        trial_dev_direction = ac;
        trial_dev_direction(trial_dev_direction==3) = trial_dev_speakers(trial_dev_direction==3);

        sal = trial_dev_speakers>-1;
        this_info = [block_cfg.target_times sal trial_dev_speakers trial_dev_direction];
        dlmwrite([path bpath 'target_info_all.txt'],this_info,'delimiter',' ');

        sal = trial_dev_speakers==1;
        this_info = [block_cfg.target_times sal trial_dev_speakers trial_dev_direction];
        dlmwrite([path bpath 'target_info_obj.txt'],this_info,'delimiter',' ');

        sal = trial_dev_direction==1;
        this_info = [block_cfg.target_times sal trial_dev_speakers trial_dev_direction];
        dlmwrite([path bpath 'target_info_dir.txt'],this_info,'delimiter',' ');
    end

end

function save_trial(stim,handl,fs,normval,path,thispath,istraining)
%%
    stim = stim/normval;
    if ~istraining
        audio_path = [path 'stim_wav/' thispath '.wav'];
        img_path = [path 'stim_records/' thispath '.jpg'];
    else
        audio_path = [path 'stim_training/' thispath '.wav'];
        img_path = [path 'stim_training/' thispath '.jpg'];
    end
    audiowrite(audio_path,stim,fs);
    r = 150; % pixels per inch
    print(handl,'-dpng',sprintf('-r%d',r),img_path); 
    close;
end

function [select_perms,select_perms_train] = get_sentences_per_trial(all_sentences,num_trials,num_train)
%% fix this function i dont even
    w = 0.5; % tolerance length difference
    sentence_perms = [];
    for s1=1:length(all_sentences{1})
        for s2=2:length(all_sentences{2})
            if abs(all_sentences{1}{s1,2}-all_sentences{2}{s2,2})<w
                sentence_perms = [sentence_perms; s1 s2 abs(all_sentences{1}{s1,2}-all_sentences{2}{s2,2})];
            end
        end
    end    
    % find the most diverse set of params to select
    %%
    select_perms = [];
    [s1_perm_counts,s1_possibilities] = hist(sentence_perms(:,1),unique(sentence_perms(:,1)));
    s2_used_counts = zeros(1,length(all_sentences{2}));
    s1_used_counts = zeros(1,length(s1_possibilities));
    for select=1:num_trials+num_train
        s1_least_used = find(s1_used_counts==min(s1_used_counts));
        [~,min_s1perm_idx] = min(s1_perm_counts(s1_least_used));
        chosen1 = s1_possibilities(s1_least_used(min_s1perm_idx(1)));
        chosens_s2s = sentence_perms(sentence_perms(:,1)==chosen1,2:3);
        min_s2perm = chosens_s2s(s2_used_counts(chosens_s2s(:,1))==min(s2_used_counts),:);
        if isempty(min_s2perm)
            chosen2 = chosens_s2s(chosens_s2s(:,2)==min(chosens_s2s(:,2)),1);
        else
            chosen2 = min_s2perm(min_s2perm(:,2)==min(min_s2perm(:,2)),1);
        end
        s1_used_counts(s1_possibilities==chosen1) = s1_used_counts(s1_possibilities==chosen1) + 1;
        s2_used_counts(chosen2) = s2_used_counts(chosen2) + 1;
        select_perms = [select_perms; chosen1 chosen2];
    end
    select_perms_train = select_perms(end-num_train+1:end,:);
    select_perms = select_perms(1:end-num_train,:);
    

end

function [trial_sentences,trial_dev_speakers,trial_dev_direction] = get_trial_info(sentence_perms,experiment_cfg)
%%
    all_speakers = experiment_cfg.all_sentences(:,2);
    dev_cases = experiment_cfg.block_cfg.dev_cases;
    dev_probs = experiment_cfg.block_cfg.dev_probs;
    num_trials = experiment_cfg.block_cfg.num_trials;    
    
    trial_sentences = [];
    perm_pointer = 1;
    for i=1:size(dev_cases,1)
        trial_sentences = [trial_sentences; repmat(dev_cases(i,:),num_trials * dev_probs(i),1)];
    end
    trial_sentences = [sentence_perms trial_sentences];
    trial_dev_speakers = trial_sentences(:,3);
%     trial_dev_speakers(trial_dev_speakers>0) = all_speakers(trial_dev_speakers(trial_dev_speakers>0));
    trial_dev_direction = categorical(trial_sentences(:,4),[-1 1 2],{'none','left','right'});
    trial_sentences = trial_sentences(:,1:2);

end

function [stim,target_time,h] = make_stim(experiment_cfg,sentence_idxs,dev_speaker,dev_direction)
%%
    % ---------------------------------------------------------------------
    % all config params ---------------------------------------------------
    % ---------------------------------------------------------------------
    
    all_sentences = experiment_cfg.all_sentences;
    fs = experiment_cfg.fs;
    analysis_len = experiment_cfg.analysis_len;
    synthesis_len = experiment_cfg.synthesis_len;
    dev_len = experiment_cfg.dev_len;
    
    d1s0 = 0; % first speaker starts from direction 0 (typically right)
    target_time = 0;
    switch_num_range = 1:5;
    switch_len = 1;
    min_stay_len = 0.8; %must be at least the same as switchlen
    min_stim_len_for_switch = arrayfun(@(num_switch)(num_switch*switch_len*2+min_stay_len*2),switch_num_range);
    
    % ---------------------------------------------------------------------
    % making of the stim (function calls mostly) --------------------------
    % ---------------------------------------------------------------------
        
    s1 = all_sentences{1}{sentence_idxs(1)};
    s2 = all_sentences{2}{sentence_idxs(2)};
    normalize_volume;
    len_stim = equalize_sentence_lengths;
    [direction_wave1,direction_wave2,t1] = make_directionality;
    hrtfs = get_hrtfs;
    add_deviant;
    len_stim = equalize_sentence_lengths;
    get_sound;
    h = show_stim(experiment_cfg,sentence_idxs,azi1,sph2nav(azi_real1),s1,azi2,sph2nav(azi_real2),s2,target_time,dev_speaker);
%     sound(stim,fs);
    disp('crimson day');
    
    % ---------------------------------------------------------------------
    % nest functions ------------------------------------------------------
    % ---------------------------------------------------------------------
    
    function normalize_volume
        s1 = s1/rms(s1);
        s2 = s2/rms(s2);
%         s1 = s1/10^(7/20);
%         s2 = s2/10^(5/20);
    end

    function len_stim = equalize_sentence_lengths
        len_stim = max(length(s1),length(s2));
        s1 = [s1; zeros(len_stim-length(s1),1)];
        s2 = [s2; zeros(len_stim-length(s2),1)];
    end

    function hrtfs = get_hrtfs
%         database='ari';       HRTFfilename='hrtf_nh172.sofa';
%         database='cipic';     HRTFfilename='subject_003.sofa';
        database='listen';    HRTFfilename='irc_1002.sofa';
        % database='mit';       HRTFfilename='mit_kemar_normal_pinna.sofa';
%         database='tu-berlin'; HRTFfilename='qu_kemar_anechoic_0.5m.sofa';
%         database='tu-berlin'; HRTFfilename='qu_kemar_anechoic_all.sofa';
        hrtfs = SOFAload(['/Users/merve/Dropbox/Grad/External tools/sofa-api-mo-1.0.1/HRTFs/SOFA/database/' database '/' HRTFfilename]);
    end

    function [direction_wave1,direction_wave2,t1] = make_directionality
        % get number of switches
        pop = switch_num_range(min_stim_len_for_switch-len_stim/fs<0);
%         switch_num = pop(randsample(length(pop),1));
        switch_num = max(pop);
        % get length of each section for s1
        extra_time = len_stim/fs-min_stim_len_for_switch(switch_num);
        section_lengths = randfixedsum(switch_num+1,1,extra_time,0,extra_time);
        % put the opening section
%         switch_wave = (1+sin(2*pi/(switch_len*2)*(0:1/fs:switch_len)-pi/2)')/2;
        switch_wave = (0:1/(fs*switch_len):1)';
        direction_wave1 = zeros(round((section_lengths(1)+min_stay_len)*fs),1);
        direction_wave2 = ones(size(direction_wave1));
        % put all of the middle sections
        for section_idx=2:switch_num
            if direction_wave1(end)==0
                this_switch = switch_wave;
                this_len = round((section_lengths(section_idx)+switch_len*2)*fs);
                direction_wave2 = [direction_wave2; ones(size(this_switch)); flip(switch_wave); zeros(this_len-length(switch_wave)*2,1); switch_wave];
            else
                this_switch = flip(switch_wave);
                this_len = round((section_lengths(section_idx)+min_stay_len)*fs);
                direction_wave2 = [direction_wave2; ones(length(switch_wave)+this_len,1)];
            end
            direction_wave1 = [direction_wave1; this_switch; ones(this_len,1)*this_switch(end)];
        end
        % put the end section
        section_idx = section_idx + 1;
        if direction_wave1(end)==0
            this_switch = switch_wave;
            this_len = round((section_lengths(section_idx)+min_stay_len)*fs);
            direction_wave2 = [direction_wave2; ones(size(this_switch)); flip(switch_wave); zeros(this_len-length(switch_wave),1)];
        else
            this_switch = flip(switch_wave);
            this_len = round((section_lengths(section_idx)+min_stay_len)*fs);
            direction_wave2 = [direction_wave2; ones(length(switch_wave)+this_len,1)];
        end
        direction_wave1 = [direction_wave1; this_switch; ones(this_len,1)*this_switch(end)];
        % checks and finalization and stuff
        if length(direction_wave1)~=length(direction_wave2)
            disp('Direction lengths not the same!');
        end
        t1 = round((min_stay_len+section_lengths(1))*fs);
        if d1s0
            direction_wave1 = direction_wave1*180-90;
            direction_wave2 = direction_wave2*180-90;
        else
            direction_wave1 = 90-direction_wave1*180;
            direction_wave2 = 90-direction_wave2*180;
        end
    end

    function add_deviant
        % initializations
        safety = round(0)*fs;
        if dev_speaker==1
            ss = s1;
            dd = direction_wave1;
            dc = circshift(dd,-round(dev_len*fs));
            tt = t1;
        elseif dev_speaker==2
            ss = s2;
            dd = direction_wave2;
            dc = circshift(dd,-round(dev_len*fs));
            tt = t1;
        end
        % selection of the target time
        while 1
            if dev_direction=='right'
                target_t = randsample(find((dd(t1:end-fs)==min(dd)).*(dc(t1:end-fs)==min(dd))),1)+t1;
            elseif dev_direction=='left'
                ffs = (dd(tt+safety:end-fs)>0).*(dc(tt+safety:end-fs )>0);
                target_t = randsample(find(ffs),1)+tt+safety;
            else 
                target_t = 0; break;
            end
            if rms(ss(target_t+(0:dev_len*fs))) > rms(ss)*0.7
                break;
            end
        end
        target_time = target_t/fs;
        % manipulating of the target section
        if dev_speaker>0
            idxs = target_t:target_t+dev_len*fs;
            seg = E_phase(ss(idxs),analysis_len,synthesis_len)';
            seg = resample(seg,analysis_len,synthesis_len);
            ss = [ss(1:target_t); seg; ss(idxs(end)+1:end)];
            if dev_speaker==1
                s1 = ss;
            elseif dev_speaker==2
                s2 = ss;
            end
        end
    end

    function get_sound
        if 0
            ear1 = s1.*direction_wave1 + s2.*direction_wave2;
            ear2 = s1.*(1-direction_wave1) + s2.*(1-direction_wave2);
            stim = [ear1, ear2];
        else
            [p1,azi1,~,sourceposition_idx1] = SOFAspat(s1,hrtfs,direction_wave1,0);
            [p2,azi2,~,sourceposition_idx2] = SOFAspat(s2,hrtfs,direction_wave2,0);
            azi_real1 = hrtfs.SourcePosition(sourceposition_idx1,1);
            azi_real2 = hrtfs.SourcePosition(sourceposition_idx2,1);
            stim = p1 + p2;
        end
    end

end

function h = show_stim(experiment_cfg,sentence_idxs,azi1,azi_real1,s1,azi2,azi_real2,s2,target_time,dev_speaker)
%%
    all_sentences = experiment_cfg.all_sentences;
    fs = experiment_cfg.fs;
    colors = ['b','r','g'];
    rect_lightness = 0.8;
    if dev_speaker==1
        dev_color = ones(1,3)*rect_lightness+[0 0 1-rect_lightness];
    elseif dev_speaker==2
        dev_color = ones(1,3)*rect_lightness+[1-rect_lightness 0 0];
    end
    
    h = figure; hold on
    yl = [-100 100];
    if target_time>0
        rectangle('Position',[target_time yl(1) experiment_cfg.dev_len diff(yl)],'FaceColor',dev_color,'linestyle','none');
    end
    
    azifactor = length(azi1)/length(s1);
    
    t = (1:length(azi1))/fs/azifactor;
    
    l1 = plot(t,azi1,'color',colors(1),'displayname',[num2str(all_sentences{1,2}) '-' num2str(sentence_idxs(1))]); 
    plot(t,azi_real1,'x','color',colors(1)); 
    l2 = plot(t,azi2,'color',colors(2),'displayname',[num2str(all_sentences{2,2}) '-' num2str(sentence_idxs(2))]);
    plot(t,azi_real2,'x','color',colors(2)); 
    
    t = (1:length(s1))/fs;
    
    plot(t,s1*5-45,'color',colors(1));
    plot(t,s2*5+45,'color',colors(2));
    
    legend([l1 l2],'location','northwest');
    ylim(yl);
    xlim([0 length(s1)/fs]);
    set(gca,'ytick',[-90 90],'yticklabel',{'right','left'});
    xlabel('Time (s)');
    
end














