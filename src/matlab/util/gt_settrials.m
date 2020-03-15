
function [eeg,varargout] = gt_settrials(fn,args,varargin)
    if iscell(args)
        eeg = args{1};
        cellargs = args(2:end);
    else
        eeg = args;
        cellargs = {};
    end

    [fnargs,params] = gt_findparams(varargin,{'progress','channels'});
    p = inputParser;
    addParameter(p,'progress','',@ischar);
    addParameter(p,'channels',[],@isnumeric);
    p.FunctionName = 'gt_fortrials';
    parse(p,params{:});

    for i = 2:nargout
        varargout{i-1} = cell(length(eeg.trial),1);
    end

    channels = p.Results.channels;
    progress = p.Results.progress;

    if ~isempty(progress)
        textprogressbar(progress);
        onCleanup(@() textprogressbar(''));
    end
    for i = 1:length(eeg.trial)
        this_cellargs = cell(length(cellargs),1);
        for a = 1:length(cellargs)
            this_cellargs{a} = cellargs{a}{i};
        end

        result = cell(nargout,1);
        if isempty(channels)
            data = eeg.trial{i}';
            [result{1:nargout}] = fn(data,this_cellargs{:},fnargs{:});
            eeg.trial{i} = result{1}';
        else
            data = eeg.trial{i};
            [result{1:nargout}] = fn(data(channels,:)',this_cellargs{:},fnargs{:});
            eeg.trial{i}(channels,:) = result{1}';
        end
        for j = 2:nargout
            varargout{j-1}{i} = result{j};
        end

        if ~isempty(progress)
            textprogressbar(100*(i/length(eeg.trial)));
        end
    end
    if ~isempty(progress)
        fprintf('\n');
    end
end
