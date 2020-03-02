function Calc_Sound_Spec_Turk(Block)

if nargin < 1
    Block = 3;
end

loadload
Folder = ['Block' num2str(Block) '/Normed/'];
% Clip_Folder = 'Normed_Clips4/'; %new folder for the wave files themselves
Out_Folder = ['Block' num2str(Block) '/Spec/']; %Normed_Specs5 - downsampled for pitch %new folder 2, before was with the option to include a DC component, now it is for removing level
% if isempty(dir(Clip_Folder))
%     mkdir(Clip_Folder)
% end
if isempty(dir(Out_Folder))
    mkdir(Out_Folder)
end



% sfs = 125;
sfs = 22050 / 176;

% for i = 1:22
%     load([Out_Folder 'Spec' num2str(i) '.mat'])
%     sfs = 22050 / 176;
%     save([Out_Folder 'Spec' num2str(i) '.mat'],'Spec','sfs')
% end

% Feat = Calc_Sal1_Features;
% nfs = 4000;
Files = dir([Folder '*.wav']);
nWav = length(Files);
nfs = 22050;
for i = 1:nWav
    [Wave,fs] = wavread([Folder 'Normed' num2str(i) '.wav']);
    if nfs ~= fs
        Wave = resample(Wave,nfs,fs);
    end
%     figure(1)
%     clf
%     subplot(2,1,1)
%     plot(Wave)
%     subplot(2,1,2)
%     Loud = Feat{i}(:,12);
%     plot(Loud,'k')
%     hold on
%     Loud = Moving_Average(Loud,32);
%     plot(Loud)
%     Loud = resample(Loud,176,1);
% 
% %     plot(Loud)
%     figure(2)
%     clf
%     subplot(2,1,1)
%     
%     Old_Max = max(abs(Wave));
%     
% %     Wave(1:3520) = [];
% %     Wave((end-3520+1):end) = [];
%     Wave = Wave ./ Loud(1:length(Wave));
%     Wave(abs(Wave) > 5) = 0;
%     Wave = Wave * Old_Max / max(abs(Wave));
%     plot(Wave)
%     subplot(2,1,2)
%     plot(Loud);
% 
% %     wavwrite(Wave,fs,[Clip_Folder 'Normed' num2str(i) '.wav']);
%     
    Spec = wav2aud(unitseq(Wave), [8 8 -2 log2(nfs/16000)]);
    save([Out_Folder 'Spec' num2str(i) '.mat'],'Spec','sfs')
end