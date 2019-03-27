function audio = sentence(stim_events,stim_info,index,source)
    global stimulus_dir;

    stim_num = stim_events{index,'sound_index'};

    % find an example sentence that isn't the same sentence as that
    % currently being used

    dont_use = stim_info.test_block_cfg.trial_sentences(stim_num,source);
    selected = 0;
    for i = randperm(size(stim_info.test_block_cfg.trial_sentences,1))
        if stim_info.test_block_cfg.trial_sentences(i,source) ~= dont_use
            selected = i;
            break;
        end
    end

    audio = audioread(fullfile(stimulus_dir,'mixtures','testing',...
        'mixture_components',sprintf('trial_%02d_%1d.wav',selected,source)));
end
