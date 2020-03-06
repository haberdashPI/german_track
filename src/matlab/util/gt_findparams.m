function [args,params] = gt_findparams(varargin,param_names)
    split = 0;
    for i = 1:length(varargin)
        if ischar(varargin{i})
            if any(strcmp(varargin{i},param_names))
                break;
            end
        end
        split = i;
    end
    if split == 0
        args = {};
        params = varargin;
    elseif split == length(varargin)
        args = varargin;
        params = {};
    else
        args = varargin(1:split);
        params = varargin((split+1):end);
    end
end
