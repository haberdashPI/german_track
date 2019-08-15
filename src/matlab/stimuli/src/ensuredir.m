
function d = ensuredir(d)
    if ~exist(d,'dir')
        mkdir(d)
    elseif exist(d) && ~exist(d,'dir')
        error(['Expected ' d ' to be a directory.'])
    end
end
