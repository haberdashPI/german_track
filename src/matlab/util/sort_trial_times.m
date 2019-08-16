function ordering = sort_trial_times(eeg,stim_events)
    function y = tonum(x)
        if strcmp(x,'test')
            y = 1;
        elseif strcmp(x,'object')
            y = 2;
        elseif strcmp(x,'feature')
            y = 3;
        end
    end

    cond_i = arrayfun(@tonum,stim_events.condition);
    [~,ordering] = sort(stim_events.sound_index + 50*(cond_i-1));
end
