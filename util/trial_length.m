function len = trial_length(fsample,stim_events,i,offset)
    global stimulus_dir

    trial_file = fullfile(stimulus_dir,"mixtures","testing",...
        sprintf("trial_%02d.wav",stim_events.sound_index(i)));

    [trial_audio,fs] = audioread(trial_file);
    audio_samples = round(size(trial_audio,1)/fs*fsample);

    len = audio_samples + round(offset*fsample);
end
