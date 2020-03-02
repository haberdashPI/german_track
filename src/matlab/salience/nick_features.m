function feats=nick_features(file_name, num_files)
feats=cell(1,num_files);
f1=textread(file_name, '%s');
for k=1:num_files
    k
    line=char(f1(k));
    [audio,fs]=audioread(line);
    audiowrite(line,resample(audio, 22050,fs), 22050);
    feats{1,k}=Calc_Salience_Features(line);
end
end