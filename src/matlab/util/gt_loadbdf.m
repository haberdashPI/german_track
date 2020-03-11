function [eeg,ntrials,markers] = gt_loadbdf(filepath,stim_events,varargin)
    head = ft_read_header(filepath);

    p = inputParser;
    addParameter(p,'channels',[],@isnumeric);
    addParameter(p,'lengths',[],@isnumeric);
    addParameter(p,'padend',0.5,@isnumeric);
    p.FunctionName = 'gt_loadbdf';
    parse(p,varargin{:});
    sound_lengths = p.Results.lengths;
    channels = p.Results.channels;
    padend = p.Results.padend;

    % define trial boundaries
    cfg = [];
    cfg.dataset = filepath;
    trial_lengths = round((sound_lengths(stim_events.sound_index)+padend)*head.Fs);
    leading = [stim_events.sample(2:end); head.nSamples];
    cfg.trl = [stim_events.sample ...
               min(leading,stim_events.sample + trial_lengths) ...
               zeros(length(stim_events.sample),1)];

    % read in the data
    cfg.continuous = 'no';
    if ~isempty(channels)
        cfg.channel = channels;
    end

    eeg = ft_preprocessing(cfg);
    ntrials = length(eeg.trial);
end
