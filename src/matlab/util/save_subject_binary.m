function save_subject_binary(eeg,filename,varargin)
    p = inputParser;
    addParameter(p,'weights',[],@iscell);
    p.FunctionName = 'save_subject_binary';
    parse(p,varargin{:})
    weights = p.Results.weights;
    label = eeg.hdr.label;

    save(filename,'eeg','weights');
end
