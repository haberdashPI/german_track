function eegfinal = gt_asfieldtrip(eeg,matrix,varargin)
    ntrials = length(eeg.trial);

    p = inputParser;
    addParameter(p,'cropfirst',0,@isnumeric);
    addParameter(p,'croplast',0,@isnumeric);
    p.FunctionName = 'gt_asfieldtrip';
    parse(p,varargin{:});
    cropfirst = p.Results.cropfirst;
    croplast = p.Results.croplast;

    eegfinal = eeg;
    eegfinal.time{1} = eegfinal.time{1}((cropfirst+1):end);
    eegfinal.time{end} = eegfinal.time{end}(1:end-croplast);
    eegfinal.trial = {};
    eegfinal.sampleinfo = eeg.sampleinfo;
    eegfinal.sampleinfo(1,1) = eegfinal.sampleinfo(1,1)+cropfirst;
    eegfinal.sampleinfo(end,2) = eegfinal.sampleinfo(end,2)-croplast;
    k = 1;
    for i = 1:ntrials
        n = size(eeg.trial{i},2);
        if i == 1
            n = n - cropfirst;
        elseif i == ntrials
            n = n - croplast;
        end
        eegfinal.trial{i} = matrix(k:(k+n-1),:)';
        k = k+n;
    end
end


