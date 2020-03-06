function [varargout] = gt_fortrials(fn,eeg,varargin)
    [fnargs,params] = gt_findparams(varargin,{'progress','channels'});
    p = inputParser;
    addParameter(p,'progress','',@ischar);
    addParameter(p,'channels',[],@isnumeric);
    p.FunctionName = 'gt_fortrials';
    parse(p,params{:});

    for i = 1:nargout
        varargout{i} = cell(length(eeg.trial),1);
    end

    channels = p.Results.channels;
    progress = p.Results.progress;

    if ~isempty(progress)
        textprogressbar(progress);
        onCleanup(@() textprogressbar(''));
    end
    for i = 1:length(eeg.trial)
        result = cell(length(nargout),1);
        data = eeg.trial{i};
        if isempty(channels)
            [result{1:nargout}] = fn(data',fnargs{:});
        else
            [result{1:nargout}] = fn(data(channels,:)',fnargs{:});
        end
        for j = 1:nargout
            varargout{j}{i} = result{j};
        end

        if ~isempty(progress)
            textprogressbar(100*(i/length(eeg.trial)));
        end
    end
    if ~isempty(progress)
        fprintf('\n');
    end
end
