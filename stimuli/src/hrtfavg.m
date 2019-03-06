readpath = '../External tools/sofa-api-mo-1.0.1/HRTFs/SOFA/database/listen/';
allfiles = dir(readpath);
allIR = zeros(187,2,512,52);
idx = 1;
for i=1:length(allfiles)
    HRTFfilename = allfiles(i).name;
    if length(HRTFfilename)<4
        continue;
    end
    hrtfs = SOFAload([readpath HRTFfilename]);
    allIR(:,:,:,idx) = hrtfs.Data.IR;
    idx=idx+1;
end
IR = mean(allIR,4);

hrtfs.Data.IR = IR;
hrtfs.GLOBAL_ListenerShortName = '1000';

save([readpath 'irc_1000.mat'],'hrtfs');