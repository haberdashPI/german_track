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

function generate_stimuli(config,block_cfg,indir,audiodata,hrtfs,is_training)
    ensuredir(indir);
    ensuredir(fullfile(indir,'target_component'));
    ensuredir(fullfile(indir,'mixture_components'));

    function saveto(str,stim,i)
        audiowrite(fullfile(indir,sprintf(str,i)),stim,config.fs);
    end

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

        saveto('trial_%02d.wav',stim,trial_idx);
        saveto(fullfile('target_component','trial_%02d.wav'),target,trial_idx);
        saveto(fullfile('mixture_components','trial_%02d_1.wav'),hrtf{1},trial_idx);
        saveto(fullfile('mixture_components','trial_%02d_2.wav'),hrtf{2},trial_idx);
        saveto(fullfile('mixture_components','trial_%02d_3.wav'),hrtf{3},trial_idx);
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
    target = block_cfg.target_speaker(trial);

    target_sound = manipulate(config,sounds{target},target_t);
    sounds{target} = target_sound;

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
