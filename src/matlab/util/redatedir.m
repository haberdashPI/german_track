function result = redatedir(p)
    d = datestr(datetime('now'),'yyyy-mm-dd_HH.MM.SS');
    [tail, ~] = fileparts(p);
    result = fullfile(tail,d);
    if ~exist(result,'dir')
        mkdir(result);
    end
end
