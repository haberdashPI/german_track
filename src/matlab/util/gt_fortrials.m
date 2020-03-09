function [varargout] = gt_fortrials(fn,args,varargin)
    if iscell(args)
        eeg = args{1};
        cellargs = args{2:end};
    else
        eeg = args;
        cellargs = cell(length(eeg.trial));
        for i = 1:length(eeg.trial)
            cellargs{i} = {};
        end
    end

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
        this_cellargs = cell(length(cellargs{i}),1);
        for a = 1:length(cellargs{i})
            this_cellargs{a} = cellargs{i}(a);
        end

        result = cell(length(nargout),1);
        data = eeg.trial{i};
        if isempty(channels)
            [result{1:nargout}] = fn(data',this_cellargs{:},fnargs{:});
        else
            [result{1:nargout}] = fn(data(channels,:)',this_cellargs{:},fnargs{:});
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
