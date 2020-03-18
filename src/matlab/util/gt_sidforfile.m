function sid = gt_sidforfile(filename)
    numstr = regexp(filename,'([0-9]+)_','tokens');
    sid = str2double(numstr{1}{1});
end
