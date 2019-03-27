function data = read_json(file)
    [fid,message] = fopen(file,'rt');
    if fid < 0
        error([message ': ' file]);
    end
    data = jsondecode(fscanf(fid,'%s'));
    fclose(fid);
end
