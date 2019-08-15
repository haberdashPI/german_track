function audio = sentence(stim_events,stim_info,stim,source)
    global stimulus_dir;

    stim_num = stim_events{stim,'sound_index'};
    audio = audioread(fullfile(stimulus_dir,'mixtures','testing',...
        'mixture_components',sprintf('trial_%02d_%1d.wav',stim_num,source)));
end
