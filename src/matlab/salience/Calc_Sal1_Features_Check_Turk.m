function [Features, Labels] = Calc_Sal1_Features_Check_Turk(Wav_List,Block,Extra,Cut_Reg)
%%%%%
% Inputs:
%%%%%
% Wav_List, which wave files to load/calc features from.  If only one
% wavefile is being used, specifying which one will save computation time.
% 
% Extra Features: except for Rand, need the exact numbers in an array
% cell structure, select from...
% {Freq} (total is 128)
% {Rate Scale} (total is 9 rate, 11 scale -> 20)
% {Ceps} (total is 64, 1 is DC)
% {MFCC} (total is 13, 1 is DC)
% {Rand} (nTrials then code (0 1 2))
%   0 for no reset, 1 for reset, 2 for perm (0 does not work with > 1 wave)
% {Bark} (total is 240)

%12/8/15 replacing HMax with flatness (18), and delta spec 1 with
%spectral irregularity (4)

%%%%%
% Initialize inputs
%%%%%

nExtra = 6; %or whatever
nWav = 20;
[Labels] = Load_Labels_Turk(cell(1,6));
nBase = length(Labels); %number of base features

if nargin < 1
    Wav_List = 1:nWav;
end
if nargin < 2
    Block = 2;
end
if nargin < 3
    Extra = cell(1,nExtra);
end
if length(Extra) < 6
    Extra{6} = [];
end
% if isempty(Extra{5})
%     Extra{5} = 0;
% end
if nargin < 4
    Cut_Reg = 0;
end

%%%%%
% Folders
%%%%%

Top = ['Block' num2str(Block)];

Audio_Dir = [Top '/Normed/'];
Spec_Dir = [Top '/Spec/'];
Cort_Dir = [Top '/Cort/'];

%%%%%
% Labels (Pull out into another function so it can be called separately)
%%%%%

Labels = Load_Labels_Turk(Extra);

%%%%%
% Other variables
%%%%%

nFeatures = length(Labels);
Features = cell(1,nWav);
Waves = cell(1,nWav);

for iWav = Wav_List
    File_Name = ['Normed' num2str(iWav) '.wav'];
    Waves{iWav} = wavread([Audio_Dir File_Name]);
end

Ceps_DF_S = 10;
Ceps_DF_L = 30;
Sal_Thresh = 1; %Pitch


Pitch_DF = 30;
Bright_DF = 30;
BW_DF = 30;

fs = 22050;
CF = cochfil(1:129,log2(fs/16000));

Extra_Index = zeros(1,nExtra);
Extra_Index(1) = nBase;
Extra_Index(2) = Extra_Index(1) + length(Extra{1});
Extra_Index(3) = Extra_Index(2) + length(Extra{2});
Extra_Index(4) = Extra_Index(3) + length(Extra{3});
Extra_Index(5) = Extra_Index(4) + length(Extra{4});
if ~isempty(Extra{5})
    Extra_Index(6) = Extra_Index(5) + Extra{5}(1);
else
    Extra_Index(6) = Extra_Index(5);
end

for iWav = Wav_List

    % Frequency bands
    load([Spec_Dir 'Spec' num2str(iWav) '.mat'])
    nTime = size(Spec,1);
    Features{iWav} = zeros(nTime,nFeatures);
    Features{iWav}(:,1) = sum(Spec(:,(CF >= 50 & CF < 300)),2) ./ sum(Spec(:,:),2); %eh.... normalized by total energy... hm
    Features{iWav}(:,2) = sum(Spec(:,(CF >= 300 & CF < 1000)),2) ./ sum(Spec(:,:),2);
    Features{iWav}(:,3) = sum(Spec(:,(CF >= 1000 & CF < 3000)),2) ./ sum(Spec(:,:),2);
    Features{iWav}(:,4) = [zeros(Pitch_DF/2,1); abs(Features{iWav}((Pitch_DF+1):end,1) - Features{iWav}(1:(end-Pitch_DF),1)) ; zeros(Pitch_DF/2,1)];
    
% replace by irregularity

    % Specific Frequencies
    if ~isempty(Extra{1})
        Features{iWav}(:,(1:length(Extra{1}))+Extra_Index(1)) = Spec(:,Extra{1});
    end
    
%     % Cepstrum, select coef
%     Ceps = Calc_Cepstrum(Spec);
%     MFCC = Calc_Cepstrum2(Waves{iWav},fs);
%     MFCC = [MFCC zeros(size(MFCC,1),1)]';
%     Features{iWav}(:,3) = Ceps(:,60); %These are random, arbitrary, and
% %     unhelpful... I think
%     Features{iWav}(:,4) = Ceps(:,11);
%     % Other Quefrencies
%     if ~isempty(Extra{3})
%         Features{iWav}(:,(1:length(Extra{3}))+Extra_Index(3)) = Ceps(:,Extra{3});
%     end
%     if ~isempty(Extra{4})
%         Features{iWav}(:,(1:length(Extra{4}))+Extra_Index(4)) = MFCC(:,Extra{4});
%     end
%     % Delta Ceps
%     Features{iWav}(:,5) = [floor(zeros(Ceps_DF_S/2,1)); abs(Features{iWav}((Ceps_DF_S+1):end,4) - Features{iWav}(1:(end-Ceps_DF_S),4)) ; ceil(zeros(Ceps_DF_S/2,1))];
%     Features{iWav}(:,6) = [floor(zeros(Ceps_DF_L/2,1)); abs(Features{iWav}((Ceps_DF_L+1):end,4) - Features{iWav}(1:(end-Ceps_DF_L),4)) ; ceil(zeros(Ceps_DF_L/2,1))];
    
    % Pitch
    temp = load([Spec_Dir 'Pitch' num2str(iWav) '.mat']);
    Features{iWav}(:,7) = temp.Pitch;
    % Delta Pitch
    Features{iWav}(:,8) = [zeros(Pitch_DF/2,1); abs(Features{iWav}((Pitch_DF+1):end,7) - Features{iWav}(1:(end-Pitch_DF),7)) ; zeros(Pitch_DF/2,1)]; %Um... fail
    % Threshold based on Saliency
    Features{iWav}(temp.Sal < Sal_Thresh,7:8) = 0;
    % Pitch Saliency
    Features{iWav}(:,5) = temp.Sal;  %Ugh, smoothing for features?
    
    %Brightness
% ORIG    Features{iWav}(:,9) = sum( Spec.^2 * diag(CF(1:(end-1))),2) ./ sum(Spec.^2,2);
    Features{iWav}(:,9) = sum( Spec * diag(CF(1:(end-1))),2) ./ sum(Spec,2);
    %Delta Brightness
    Features{iWav}(:,10) = [zeros(Bright_DF/2,1); abs(Features{iWav}((Bright_DF+1):end,9) - Features{iWav}(1:(end-Bright_DF),9)) ; zeros(Bright_DF/2,1)];
    %Bandwidth
    for iTime = 1:nTime
%         Features{iWav}(iTime,11) = (CF(1:(end-1)) - Features{iWav}(iTime,9)) .^2 * Spec(iTime,:)' / sum(Spec(iTime,:).^2);
% ORIG        Features{iWav}(iTime,11) = sqrt((CF(1:(end-1)) - Features{iWav}(iTime,9)) .^2 * Spec(iTime,:).^2' / sum(Spec(iTime,:).^2));
        Features{iWav}(iTime,11) = (CF(1:(end-1)) - Features{iWav}(iTime,9)) * Spec(iTime,:)' / sum(Spec(iTime,:));
    end
    %To Delta BW?

%     Features{iWav}(:,11) = [zeros(BW_DF/2,1); abs(Features{iWav}((BW_DF+1):end,11) - Features{iWav}(1:(end-BW_DF),11)) ; zeros(BW_DF/2,1)];
    %Spectral Flatness
    Features{iWav}(:,18) = prod(Spec,2).^(1/size(Spec,2)) ./ mean(Spec,2);
    %Spectral Irregularity
    Features{iWav}(:,4) = sum((Spec(:,2:end) - Spec(:,1:(end-1))).^2,2) ./ sum(Spec.^2,2);
    1;
    
end

%Load Data
load([Top '/Loud_Zero.mat'])
load([Top '/Loud_Sharp_NoBark.mat'])
% load('Spec_SVD.mat') %Unused currently

for iWav = Wav_List
    
    nTime_Loud = length(Loud{iWav});

    %Loudness
    Features{iWav}(:,12) = Loudness{iWav};
    %Zero Crossings
    Features{iWav}(:,13) = Zero_Cross{iWav};
    Features{iWav}(1:nTime_Loud,19) = Loud{iWav};
    Features{iWav}(1:nTime_Loud,20) = Sharp{iWav};
    
    %Loudness in bark bands
    if ~isempty(Extra{6})
        Features{iWav}(1:nTime_Loud,(1:length(Extra{6}))+Extra_Index(6)) = Loud_Bark{iWav}(:,Extra{6});
    end
    
    Features{iWav}(:,6) = [zeros(Pitch_DF/2,1); abs(Features{iWav}((Pitch_DF+1):end,19) - Features{iWav}(1:(end-Pitch_DF),19)) ; zeros(Pitch_DF/2,1)];
    
    % Some others
    % Features{iWav}(:,10) = First_SVD{iWav};
    % Features{iWav}(:,20) = log10(Loudness{iWav});
end


for iWav = Wav_List

    % Cortical features
    load([Cort_Dir 'Cort' num2str(iWav) '.mat'])
    % Minor processing
    Cort_Rate = abs(Cort_Rate);
    Cort_Rate = Cort_Rate(:,1:9) + Cort_Rate(:,10:18);
    Cort_Scale = abs(Cort_Scale);
    % Rate
    Features{iWav}(:,14) = mean(Cort_Rate(:,1:4),2);
    Features{iWav}(:,15) = mean(Cort_Rate(:,5:9),2);
    % Scale
    Features{iWav}(:,16) = mean(Cort_Scale(:,1:5),2);
    Features{iWav}(:,17) = mean(Cort_Scale(:,6:11),2);

    % Individual
    if ~isempty(Extra{2})
        Cort_Temp = [Cort_Rate Cort_Scale];
        Features{iWav}(:,(1:length(Extra{2}))+Extra_Index(2)) = Cort_Temp(:,Extra{2});
    end
end

% %%%%%
% %HMax response
% %%%%%
% 
% [~, ~, Tag] = set_HMax_Params;
% Tag = 'Limit_NoNorm_SmFilt_AllCort_';
% MaxC1 = Load_Sal1_HMax_Response(Tag,[],Wav_List);
% 
% for iWav = Wav_List
%     %ONE MWA MAXC1
%     M = max(MaxC1{iWav},[],1) / max(max(MaxC1{iWav}));
%     M = Moving_Average(M,24); %May remove MWA
%     M = resample(M,8,1)';
% 
%     mLength = min(size(Features{iWav},1),length(M));
%     Features{iWav}(1:mLength,18) = M(1:mLength);
% end


% Remove all normal labels
if Cut_Reg
    Labels(1:20) = [];
    for iWav = Wav_List
        Features{iWav}(:,1:20) = [];
    end
end

%Random feature
if ~isempty(Extra{5})
    nTrial = Extra{5}(1);
    if Extra{5}(2) == 2
        load('C:\Users\Guest\Documents\Random_Trials.mat')
        for iWav = Wav_List
            Features{iWav}(:,(1:nTrial)+Extra_Index(5)) = RNG{iWav}(:,1:nTrial);
        end
    else
        %Note, currently this does not function with > 1 wav
        for iWav = Wav_List
            if Extra{5}(2) == 1
                nTime = size(Features{iWav},1);
                RNG = abs(randn(nTime,nTrial));
                save('Rand_Temp.mat','RNG','-v7.3')
            end
            load('Rand_Temp.mat')

            Features{iWav}(:,(1:nTrial)+Extra_Index(5)) = RNG(:,1:nTrial);
        end
    end
end

% Adjust for noise at front and back
for iWav = Wav_List
    Features{iWav}([1:20 (end-19):end],:) = [];
end

% Corr between features... maybe useful?
% Corr_Matrix = zeros(nFeatures,nFeatures);
% for iFeat = 1:(nFeatures-1)
%     for jFeat = (i+1):nFeatures
%         for iWav = Wav_List
%             Corr_Matrix(iFeat,jFeat) = Corr_Matrix(iFeat,jFeat) + c
%         end
%     end
% end