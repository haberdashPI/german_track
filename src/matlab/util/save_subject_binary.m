function save_subject_binary(subj,filename,varargin)
    p = inputParser;
    addParameter(p,'weights',[],@iscell);
    p.FunctionName = 'save_subject_binary';
    parse(p,varargin{:})
    weights = p.Results.weights;

    fid = fopen(filename,'w','n','UTF-8');
    try
        % number of channels
        nchan = size(subj.trial{1},1);
        fwrite(fid,2,'int32'); % file version
        fwrite(fid,~isempty(weights),'uint8');
        fwrite(fid,nchan,'int32');
        % fprintf('number of channels %d\n',nchan);
        % channel names
        for i = 1:size(subj.trial{1},1)
            nchar = numel(subj.hdr.label{i});
            % fprintf('number of chars %d\n',nchar);
            fwrite(fid,nchar,'int32');
            fwrite(fid,subj.hdr.label{i},'char');
        end
        % number of trials
        ntrial = length(subj.trial);
        % fprintf('number of trials %d\n',ntrial)
        fwrite(fid,ntrial,'int32');
        % sample rate
        % fprintf('sample rate %d\n',subj.fsample)
        fwrite(fid,subj.fsample,'int32');
        % trials
        for i = 1:length(subj.trial)
            % size of trial
            trial = subj.trial{i};
            trial_size = size(trial);
            % fprintf('Trial size: %d %d\n',trial_size)
            fwrite(fid,trial_size,'int32');
            fwrite(fid,trial,'float64');
            if ~isempty(weights)
                if all(size(weights{i}) ~= size(subj.trial{i}))
                    error("Weight size does not match trial size.")
                end
                fwrite(fid,weights{i},'float64');
            end
        end
    catch e
        fclose(fid);
        rethrow(e);
    end
end
