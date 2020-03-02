function Calc_Spec_Cort_Turk(Block)
    
if nargin < 1
    Block = 3;
end

loadload
Folder = ['Block' num2str(Block) '/Spec/'];
% Out_Folder = 'C:/Users/Guest/Documents/Wav_Cort/';
Out_Folder = ['Block' num2str(Block) '/Cort/']; %Normed_Cort2 was with DC... hm, now it is after removing level
if isempty(dir(Out_Folder))
    mkdir(Out_Folder)
end

% sfs = 125;
% sfs = 22050 / 176;
fs = 22050;

nWav = length(dir([Folder '*.mat']));

for i = 1:nWav
%     [Wave,fs] = wavread([Folder 'Normed' num2str(i) '.wav']);
%     Wave = resample(Wave,16000,fs);
%     Spec = wav2aud(unitseq(Wave), [8 8 -2 log2(fs/16000)]);
    load([Folder 'Spec' num2str(i) '.mat'])
    
    Cort = aud2cor(Spec, [8 8 -2 log2(fs/16000) 0 0 1], 2.^(1:.5:5), 2.^(-2:.5:3),'tmpxxx',0);
    Cort_Rate = squeeze(mean(mean(Cort,1),4))';
    Cort_Scale = squeeze(mean(mean(Cort,2),4))';
    Cort_Freq = squeeze(mean(mean(Cort,1),2))';
    clear Cort
    
%     Cort = aud2cor(Spec, [8 8 -2 log2(fs/16000) 0 0 1], 2.^(3:.25:7), 2.^(-2:3),'tmpxxx',0);
%     Cort_Rate = squeeze(mean(mean(Cort,1),4))';
%     clear Cort
%     
%     Cort = aud2cor(Spec, [8 8 -2 log2(fs/16000) 0 0 1], 2.^(3:7), 2.^(-2:.25:3),'tmpxxx',0);
%     Cort_Scale = squeeze(mean(mean(Cort,2),4))';
%     clear Cort

%     Cort_Freq = squeeze(mean(mean(Cort,1),2));

    save([Out_Folder 'Cort' num2str(i) '.mat'], 'Cort_Rate', 'Cort_Scale','Cort_Freq','sfs','-v7.3');
    
    
%     Cort_Rate = squeeze(mean(mean(Cort,1),4))';
%     Cort_Scale = squeeze(mean(mean(Cort,2),4))';
%     Cort_Freq = squeeze(mean(mean(Cort,1),2));

%     save([Out_Folder 'Cort' num2str(i) '.mat'], 'Cort','sfs','-v7.3');
end