
function eeg = gt_settrials(fn,args,varargin)
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

        if isempty(channels)
            data = eeg.trial{i}';
            eeg.trial{i} = fn(data,this_cellargs{:},fnargs{:})';
        else
            data = eeg.trial{i};
            data(channels,:) = fn(data(channels,:)',this_cellargs{:},fnargs{:})';
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
