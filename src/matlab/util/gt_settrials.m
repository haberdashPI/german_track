
function eeg = gt_settrials(fn,eeg,varargin)
    [fnargs,params] = gt_findparams(varargin,{'progress','channels'});
    p = inputParser;
    addParameter(p,'progress','',@ischar);
    addParameter(p,'channels',[],@isnumeric);
    p.FunctionName = 'gt_fortrials';
    parse(p,params{:});

    channels = p.Results.channels;
    progress = p.Results.progress;

    if ~isempty(progress)
        textprogressbar(progress);
        onCleanup(@() textprogressbar(''));
    end
    for i = 1:length(eeg.trial)
        if isempty(channels)
            data = eeg.trial{i}';
            eeg.trial{i} = fn(data,fnargs{:})';
        else
            data = eeg.trial{i};
            data(channels,:) = fn(data(channels,:)',fnargs{:})';
            eeg.trial{i} = data;
        end

        if ~isempty(progress)
            textprogressbar(100*(i/length(eeg.trial)));
        end
    end
    if ~isempty(progress)
        fprintf('\n');
    end
end
