function Features=Calc_Salience_Features(File_Name)

loadload; %requires NSL tools

if nargin < 1
    
    File_Name = 'Test.wav';
    
end

Full_File = File_Name;

Divs = strfind(File_Name,'/');
if isempty(Divs)
    Folder = './';
else
    Divs = Divs(end);
    Folder = File_Name(1:Divs);
    File_Name = File_Name((Divs+1):end);
end

Storage = [Folder 'tmp_Features/'];
if ~isdir(Storage)
    mkdir(Storage)
end

File_Name = strtok(File_Name,'.');
%Mat_File = [Folder File_Name '.mat'];



%first, perform some calculations if not already done

Spec_File = [Storage File_Name '_Spec.mat'];
Cort_File = [Storage File_Name '_Cort.mat'];
Pitch_File = [Storage File_Name '_Pitch.mat'];
Loud_File = [Storage File_Name '_Loud.mat'];

[Wave,fs] = audioread(Full_File);

CF = cochfil(1:129,log2(fs/16000));

%Spectrogram
if isempty(dir(Spec_File));
    
    sfs = fs / 176;
    
    Spec = wav2aud(unitseq(Wave), [8 8 -2 log2(fs/16000)]);
    save(Spec_File,'Spec','sfs')
    
    1;
else
    load(Spec_File);
end

%Cortical representation
if isempty(dir(Cort_File));
    
    Cort = aud2cor(Spec, [8 8 -2 log2(fs/16000) 0 0 1], 2.^(1:1:5), 2.^(-2:1:3),'tmpxxx',0);
    Cort_Rate = squeeze(mean(mean(Cort,1),4))';
    Cort_Scale = squeeze(mean(mean(Cort,2),4))';
    Cort_Freq = squeeze(mean(mean(Cort,1),2))';
    clear Cort
    
    save(Cort_File, 'Cort_Rate', 'Cort_Scale','Cort_Freq','sfs','-v7.3');
    
    1;
else
    load(Cort_File)
end

%Pitch and harmonicity
if isempty(dir(Pitch_File));
    
    
    th = exp(mean(log(max(Spec(:),1e-3))));
    [Pitch,Sal] = pitch(log(max(Spec',th)) - log(th), CF(1:(end-1)),'pitlet_templates');
    save(Pitch_File,'Pitch','Sal')
    
    1;
else
    load(Pitch_File)
end

%Loudness and Sharpness
if isempty(dir(Loud_File));
    
    
    Window = 1500;
    Step = 240;
    lfs = 30000;
    
    Sound = resample(Wave,lfs,fs); %hm...
    nTime = ceil(length(Sound) / Step);
    Sound(ceil(nTime * Step)) = 0;
    
    Loud = zeros(nTime,1);
    Sharp = zeros(nTime,1);
    Loud_Bark = zeros(nTime,240);
    Loud_P = zeros(nTime,21);
    
    Off = ceil(Window / Step / 2)+1;
    
    %     Points = .008 * 30000; %consider stepping
    %     nTime = floor(length(Sound) / Points);
    %     N_entire = zeros(1,nTime);
    for iTime = Off:(nTime - Off + 1)
        Sample = (iTime - 1) * Step;
        x = Sound((Sample - Window / 2):(Sample + Window / 2));
        [Loud(iTime),N_Single,P] = loudness_1991(x,60,lfs,0);
        Loud_Bark(iTime,:) = N_Single;
        Sharp(iTime) = sharpness_Fastl(N_Single);
        Loud_P(iTime,:) = P;
    end
    
    Sharp_Zero = Sharp;
    Fill = isnan(Sharp);
    [First,Last] = First_and_Last(find(Fill));
    for iFirst = 1:length(First)
        F = First(iFirst) - 1;
        L = Last(iFirst) + 1;
        Sharp(F:L) = linspace(Sharp(F),Sharp(L),L - F + 1);
    end
    Sharp_Zero(isnan(Sharp_Zero)) = 0;
    
    save(Loud_File,'Loud','Sharp','Loud_Bark','Sharp_Zero','Loud_P','lfs');
    %     save(['Block' num2str(Block) '/Loud_Sharp_NoBark.mat'],'Loud','Sharp','Sharp_Zero','-v7.3');
    1;
else
    load(Loud_File)
end

%Brightness
Bright = sum( Spec * diag(CF(1:(end-1))),2) ./ sum(Spec,2);

%Bandwidth
nTime = size(Spec,1);
BW = zeros(size(Bright));
for iTime = 1:nTime
    BW(iTime) = (CF(1:(end-1)) - Bright(iTime)) * Spec(iTime,:)' / sum(Spec(iTime,:));
end

%Spectral Flatness
Flat = prod(Spec,2).^(1/size(Spec,2)) ./ mean(Spec,2);

%Spectral Irregularity
Irregularity = sum((Spec(:,2:end) - Spec(:,1:(end-1))).^2,2) ./ sum(Spec.^2,2);

%Edge effects
temp = [Bright BW Flat Irregularity];
temp(1:20,:) = 0;
temp((end-19):end,:) = 0;

%Loudness adjustment
Loud_Bark = [Loud_Bark; zeros(size(Spec,1) - size(Loud_Bark,1),size(Loud_Bark,2))];
Loud_Bark(isnan(Loud_Bark)) = 0;


temp = [temp Pitch Sal];
Single_Features = Moving_Average(temp',128);
Single_Features = Single_Features';
%change first and last 20 to zeros?

Cort_Rate = abs(Cort_Rate);
Cort_Rate(:,1:6) = fliplr(Cort_Rate(:,1:6));
Cort_Scale = abs(Cort_Scale);


%Convert to one dimension

Window = 32;
% iEvent = 3;
% Window = 1280;
Step = Window / 4;


Points = floor(nTime / Step - Window / Step);
C = zeros(Points,15);

RV = 2.^(1:1:5);
RV = [fliplr(-(2.^(1:1:5))) 2.^(1:1:5)];
SV = 2.^(-2:1:3);

%so this is basically just taking another moving average, except for Rate
%and Scale
for iTime = (Step+2):(Points-1)
    Samp = (iTime)*Step + (1:Window);
    
    %Single features, vector matches the previous order
    C(iTime+1,[7 4 9 10 6 8]) = mean(Single_Features(Samp,:),1);
    
    %Spec average energy
    Stim = Spec(Samp,:);
    Mean_Stim = mean(Stim,1)';
    C(iTime+1,1) = mean(Mean_Stim);
    
    %Rate, average energy
    Stim = Cort_Rate(Samp,:);
    Mean_Stim = mean(Stim,1)';
    C(iTime+1,2) = mean(Mean_Stim);
    
    %Scale, average energy
    Stim = Cort_Scale(Samp,:);
    Mean_Stim = mean(Stim,1)';
    C(iTime+1,3) = mean(Mean_Stim);
% 
    %Spec average energy
    Stim = Loud_Bark(Samp,:);
    Mean_Stim = mean(Stim,1)';
    C(iTime+1,5) = mean(Mean_Stim);
%     
%         
    %Rate, Max
    Stim = Cort_Rate(Samp,:);
    Max_Stim = max(Stim,[],1)'; 
    C(iTime+1,11) = max(Max_Stim); 
% 
    %Scale, Max
    Stim = Cort_Scale(Samp,:);
    Max_Stim = max(Stim,[],1)'; 
    C(iTime+1,12) = max(Max_Stim); 
% 
    %Rate, Centroid
    Stim = Cort_Rate(Samp,:);
    Cent_Stim = sum( Stim.^2 * diag(RV),2) ./ sum(Stim.^2,2);
    C(iTime+1,13) = mean(Cent_Stim); 

    %Scale, Centroid
    Stim = Cort_Scale(Samp,:);
    Cent_Stim = sum( Stim.^2 * diag(SV),2) ./ sum(Stim.^2,2);
    C(iTime+1,14) = mean(Cent_Stim); 

%    Rate, Centroid using absolute value of rate
    Stim = Cort_Rate(Samp,:);
    Cent_Stim = sum( Stim.^2 * diag(abs(RV)),2) ./ sum(Stim.^2,2);
    C(iTime+1,15) = mean(Cent_Stim);
    
end

M = mean(C,1);
S = std(C,[],1);
C = (C - repmat(M,[size(C,1) 1])) ./ repmat(S,[size(C,1) 1]);
Features = C;
1;


function [Output] = Moving_Average(Input,Window)


%041615 allowing matrices, hopefully it's faster for the eeg chans

%Adjust to make sure it's a row vector if it's a single dimensional
%(because I don't think I was consistent there).

Flip_Flag = 0;
if size(Input,2) == 1;
    Input = Input';
    Flip_Flag = 1;
end
% if size(Input,1) == 1;
%     Flip_Flag = 1;
% end

if Window == 1;
    Output = Input;
    return
end
Output = zeros(size(Input));
E = size(Output,2);

for iTime = 1:(Window/2)
    Output(:,iTime) = mean(Input(:,1:(iTime + Window/2)),2);
    Output(:,E - iTime+1) = mean(Input(:,(E - iTime+1 - Window/2):E),2);
end
%Actually, window is window + 1...
for iTime = (Window/2 + 1):(E - Window / 2)
    Output(:,iTime) = mean(Input(:,(iTime - Window/2):(iTime + Window/2)),2);
end

if Flip_Flag
    Output = Output';
end

function [First,Last] = First_and_Last(Input)

%Input is in the form of find(logical)
%Output is first indices and last indices

First = Input - 1;
First = First(~ismember(First,Input)) + 1;
Last = Input + 1;
Last = Last(~ismember(Last,Input)) - 1;