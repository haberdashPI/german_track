function create_mixtures(indir)

    % GOOD GOD: I really wish people would write *modular* functions.
    % Non-local relationships between variables are abundant here. So.
    % Confusing.  I would love to clean this code up more, but it is simply not
    % worth my time.

    % NOTE: no audio files will be saved in the config file, since these will
    % be generated as *.wav files.

    config_file = fullfile(indir,'config.json');
    fid = fopen(config_file);
    config = jsondecode(fscanf(fid,'%s'));
    fclose(fid);
    disp(['Read configuration from "' config_file '".']);

    hrtfs = SOFAload(fullfile(indir,'hrtfs','hrtf_b_nh172.sofa'));

    [audiodata,fs] = read_sentences(fullfile(indir,config.sentence_dir),...
        config.speaker_order);
    if fs ~= config.fs
        error(['Sample rates of audio files (' num2str(fs) ...
               ') and configuration (' num2str(config.fs) ') do not match.']);
    end

    disp('Training stimuli:');
    generate_stimuli(config,config.train_block_cfg,...
        fullfile(indir,config.mix_dir,'training'),audiodata,hrtfs,1);
    disp('Testing stimuli:');
    generate_stimuli(config,config.test_block_cfg,...
        fullfile(indir,config.mix_dir,'testing'),audiodata,hrtfs,0);
end

function generate_stimuli(config,block_cfg,indir,audiodata,hrtfs,is_training)
    ensuredir(indir);
    ensuredir(fullfile(indir,'target_component'));
    ensuredir(fullfile(indir,'mixture_components'));

    delete(fullfile(indir,'*.wav'));
    delete(fullfile(indir,'target_component','*.wav'));
    delete(fullfile(indir,'mixture_components','*.wav'));

    function saveto(str,stim,i)
        audiowrite(fullfile(indir,sprintf(str,i)),stim,config.fs);
    end

    warning('error','MATLAB:audiovideo:audiowrite:dataClipped');
    textprogressbar('Generating mixtures...');
    onCleanup(@() textprogressbar('\n'));
    bad_trials = [];
    for trial_idx=1:block_cfg.num_trials
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

        try
            saveto('trial_%02d.wav',stim,trial_idx);
            if ~isempty(target)
                saveto(fullfile('target_component','trial_%02d.wav'),target,trial_idx);
            end
            saveto(fullfile('mixture_components','trial_%02d_1.wav'),hrtf{1},trial_idx);
            saveto(fullfile('mixture_components','trial_%02d_2.wav'),hrtf{2},trial_idx);
            saveto(fullfile('mixture_components','trial_%02d_3.wav'),hrtf{3},trial_idx);
        catch e
            switch e.identifier
            case 'MATLAB:audiovideo:audiowrite:dataClipped'
                bad_trials = [bad_trials trial_idx];
            otherwise
                rethrow(e);
            end
        end

        textprogressbar(100*(trial_idx/block_cfg.num_trials));
    end

    save_target_info(block_cfg,indir)

    if ~isempty(bad_trials)
        warning(['Some of the trials had clipped audio: ' num2str(bad_trials)])
    end
end

function save_target_info(block_cfg,indir)
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
    dlmwrite(fullfile(indir,'target_info_all.txt'),this_info,'delimiter',' ');

    sal = trial_target_speakers==1;
    this_info = [block_cfg.target_times sal trial_target_speakers trial_target_dir];
    dlmwrite(fullfile(indir,'target_info_obj.txt'),this_info,'delimiter',' ');

    sal = trial_target_dir==1;
    this_info = [block_cfg.target_times sal trial_target_speakers trial_target_dir];
    dlmwrite(fullfile(indir,'target_info_dir.txt'),this_info,'delimiter',' ');
end

function [stim,target_stim,hrtf_stims] = make_stim(config,block_cfg,trial,...
    audiodata,hrtfs,loud_target)

    % get the targets
    idxs = block_cfg.trial_sentences(trial,:);
    s1 = audiodata{1}(idxs(1)).data;
    s2 = audiodata{2}(idxs(2)).data;
    s3 = audiodata{3}(idxs(3)).data;

    % insert the target sound
    [target_stim,sounds] = add_target(config,block_cfg,trial,{s1,s2,s3},...
        loud_target);
    [~,s1,s2,s3] = equalize_lengths(sounds{1},sounds{2},sounds{3});

    % compute directions of the sounds
    direc = make_directions(config,block_cfg,trial,audiodata);

    % transform sounds by HRTFs
    p1 = SOFAspat(s1,hrtfs,direc{1},0);
    p2 = SOFAspat(s2,hrtfs,direc{2},0);
    p3 = SOFAspat(s3,hrtfs,direc{3},0);
    hrtf_stims = {p1,p2,p3};

    % mix the sounds
    stim = p1 + p2 + p3;
end

function [target_sound,sounds] = add_target(config,block_cfg,trial,sounds,loud_target)
    target_t = block_cfg.target_indices(trial);
    target_speaker = block_cfg.trial_target_speakers(trial);
    if target_speaker < 0
        target_sound = [];
        return;
    end

    target_sound = manipulate(config,sounds{target_speaker},target_t);
    sounds{target_speaker} = target_sound;

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
