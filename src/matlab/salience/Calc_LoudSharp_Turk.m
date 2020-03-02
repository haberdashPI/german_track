function Calc_LoudSharp_Turk(Block)

if nargin < 1
    Block = 2;
end
Audio_Dir = ['Block' num2str(Block) '\Normed\'];

nWav = length(dir([Audio_Dir 'Normed*.wav']));

    
Window = 1500;
Step = 240;
fs = 30000;


Loud = cell(1,20);
Loud_Bark = cell(1,20);
Loud_P = cell(1,20);
Sharp = cell(1,20);

for iWav = 1:20
    
    Sound = wavread([Audio_Dir '/Normed' num2str(iWav) '.wav']);
%     Sound = wavread(['EEG_Stim/Mix' num2str(iWav) '.wav']);
    
    Sound = resample(Sound,200,147);
    nTime = ceil(length(Sound) / Step);
    Sound(ceil(nTime * Step)) = 0;
    
    Loud{iWav} = zeros(nTime,1);
    Sharp{iWav} = zeros(nTime,1);
    Loud_Bark{iWav} = zeros(nTime,240);
    Loud_P{iWav} = zeros(nTime,21);
    
    Off = ceil(Window / Step / 2)+1;
    
%     Points = .008 * 30000; %consider stepping
%     nTime = floor(length(Sound) / Points);
%     N_entire = zeros(1,nTime);
    tic
    for iTime = Off:(nTime - Off + 1)
        Sample = (iTime - 1) * Step;
        x = Sound((Sample - Window / 2):(Sample + Window / 2));
        [Loud{iWav}(iTime),N_Single,P] = loudness_1991(x,60,fs,0);
        Loud_Bark{iWav}(iTime,:) = N_Single;
        Sharp{iWav}(iTime) = sharpness_Fastl(N_Single);
        Loud_P{iWav}(iTime,:) = P;
    end
    toc
    1;
%     Sharp{iWav}(isnan(Sharp{iWav})) = 0;
%     figure(1)
%     plot(Sharp{iWav})
%     1;
    
end

save(['Block' num2str(Block) '/Loud_Sharp.mat'],'Loud','Sharp','Loud_Bark','Loud_P');



%Fill isnan sections of Sharp %frak, I loaded the wrong file for
%calculating loudness, may affect the rev corr calculations
load(['Block' num2str(Block) '/Loud_Sharp.mat']);
Sharp_Zero = cell(1,20);
for iWav = 1:20
    Sharp_Zero{iWav} = Sharp{iWav};
    Fill = isnan(Sharp{iWav});
    [First,Last] = First_and_Last(find(Fill));
    for iFirst = 1:length(First)
        F = First(iFirst) - 1;
        L = Last(iFirst) + 1;
        Sharp{iWav}(F:L) = linspace(Sharp{iWav}(F),Sharp{iWav}(L),L - F + 1);
    end
    Sharp_Zero{iWav}(isnan(Sharp_Zero{iWav})) = 0;
%     figure(3)
%     clf
%     plot(Sharp{iWav})
%     hold on
%     plot(Hold,'r')
%     pause
end
save(['Block' num2str(Block) '/Loud_Sharp.mat'],'Loud','Sharp','Loud_Bark','Sharp_Zero','Loud_P');
% load('Loud_Sharp_Mix.mat')
save(['Block' num2str(Block) '/Loud_Sharp_NoBark.mat'],'Loud','Sharp','Sharp_Zero','-v7.3');