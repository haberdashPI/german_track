function write_json(file,data)
    fid = fopen(file,'wt');
    fprintf(fid,'%s',jsonencode(data));
    fclose(fid);
end
