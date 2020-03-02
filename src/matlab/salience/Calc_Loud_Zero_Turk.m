function Calc_Loud_Zero_Turk(Block)
    
if nargin < 1
    Block = 2;
end
Audio_Dir = ['Block' num2str(Block) '\Normed\'];

nWav = length(dir([Audio_Dir 'Normed*.wav']));


% nWav = 20;
Loudness = cell(1,nWav);
Zero_Cross = cell(1,nWav);
Window_Length = 1000;
Window_Step = 176;
% Window_Step = 128;

lfs = 22050 / Window_Step;

for iWav = 1:nWav
%     for iWav = 1:nWav
    File_Name = ['Normed' num2str(iWav) '.wav'];
    Wave = wavread([Audio_Dir File_Name]);
%     Wave = resample(Wave,16000, 22050);
    Points = ceil(length(Wave) / Window_Step);
    Wave(ceil(Points * Window_Step)) = 0; %Zero pad
    Loudness{iWav} = zeros(Points,1);
    Zero_Cross{iWav} = zeros(Points,1);
    Off = ceil(Window_Length / Window_Step / 2)+1;
    Zero_Wav = sign(Wave);
    [Zero_First, Zero_Last] = First_and_Last(find(Zero_Wav == 1));
    
    for iTime = Off:(Points - Off + 1)
        Sample = (iTime - 1)*Window_Step;
        Loudness{iWav}(iTime) = rms( Wave((Sample - Window_Length/2):(Sample + Window_Length/2)) );
        Zero_Cross{iWav}(iTime) = sum(Zero_First < Sample + Window_Length & Zero_First > Sample - Window_Length) + sum(Zero_Last < Sample + Window_Length & Zero_Last > Sample - Window_Length);
%         Zero_Cross{iWav}(iTime) = ;
    end
        
end
save(['Block' num2str(Block) '\Loud_Zero.mat'],'Loudness','Zero_Cross','lfs');