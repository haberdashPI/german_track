function [trial,label,w] = load_subject_binary(filename)
    data = load(filename);
    if isfield(data,'weights')
        w = data.weights;
    end
    trial = data.eeg.trial;
    label = data.eeg.hdr.label;
end
