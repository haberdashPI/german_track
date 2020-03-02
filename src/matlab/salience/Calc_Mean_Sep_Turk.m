function [Output,Output2] = Calc_Mean_Sep_Turk(Block)
    
if nargin < 1
    Block = 4;
end

%Sep will be separate events for each type of input.  Of course, there will
%be no Alt, because it will technically be Alt by default.  Combining
%events will be dealt with in Calc_Mem_Sep_Events or something like that.
% close all

%ExpMem_Sep3 -> try adding high pass filter?
%ExpMem is an outdated 'model', best performance was just max energy in
%frequency bands, where the energy is averaged in time windows.  MaxMean

% This program just saves the output I believe, doesn't calculate events
% yet.  Events calculated in Calc_Mean_PrePost? meh
MFile = 'Calc_Mean_Sep_Turk.m';
% MatFile = 'Mean_Sep_1D_HR_NoNorm.mat';
MatFile = ['Block' num2str(Block) '/Mean_Sep_Turk'];

%LoudOvBW, removing BW and replace with loudness, 

%NoNorm, removing fnorm in Load_Spec_Plus_1D, please remember to change it
%back





A = dir(MFile);
B = dir(MatFile);
if ~isempty(dir(MatFile))
    
    if datenum(B.date) > datenum(A.date);
        load(MatFile);
        return
    end
end


%... ........
% what about...
% find the highest value of max minus min in the same channel?
% yeah, I'd probably get some issue with normalization...
% Small windows I guess.  Consider some smoothing first too.

%has to be 2D cort, doens't make much sense otherwise...

% [Window,Step,Tag] = set_HMax_Params;
Window = 32;
% iEvent = 3;
% Window = 1280;
Step = Window / 4;
% Step = 1; %Trying more resolution for Merve's Model? %AH, here
% Resp = Load_Sal1_Response;
% Wav_List = 1:20;
% nWav = length(Wav_List);
% MaxC1 = cell(1,nWav);
% MeanC1 = cell(1,nWav);
% MaxC2 = cell(1,nWav);
% MeanC2 = cell(1,nWav);

fs = 100;
Dur = 10; %sec
Tau = 2; %sec

Mem = exp(-(0:1/fs:Dur)/Tau);
Mem = fliplr(Mem);

Mem = ones(size(Mem));
Mem = Mem' / sum(Mem);
nMem = length(Mem);
% 
% figure(1)
% plot(Mem)
% sum(Mem)
Output = cell(1,20);
Output2 = cell(2,20);
Freq = cell(1,20);


% cr(sdx, rdx+(sgn==1)*K1, :, :) = z; %This was the format for rate and
% scale.  Rate goes from neg low to high, then pos low to high.  Rate and Scale vectors afterwards.

RV = 2.^(1:.5:5);
RV = [fliplr(-(2.^(1:.5:5))) 2.^(1:.5:5)];
SV = 2.^(-2:.5:3);

Vec{2} = RV;
Vec{3} = SV;

% X = 1:Window;
nWav = length(dir(['Block' num2str(Block) '\Normed\Normed*.wav']));
for iWav = 1:nWav
% %     Wav_List(iWav)
%     
%     load(['Normed_Cort\Cort' num2str(Wav_List(iWav)) '.mat'])
% %     Cort_Freq = abs(Cort_Freq');
%     load(['Normed_Specs\Spec' num2str(Wav_List(iWav)) '.mat'])
%     Cort_Freq = Spec;
% 
% 
% %     load(['C:\Users\Guest.LCAP-HACKERMANL\Documents\Wav_2DCort\2DCort' num2str(Wav_List(iWav)) '.mat']);
% 
%     Cort_Rate = abs(Cort_Rate);
%     Cort_Scale = abs(Cort_Scale);

% %     C = zeros(Points,1);
% %     D = zeros(Points,1);
% 
% %     Ind = zeros(Points,1);
%     %Maybe some norm within channels...
%     %oops, I forgot to try it
%     % do it now
%     
%     
%     nFreq = size(Cort_Freq,2);
%     nRate = size(Cort_Rate,2);
%     nScale = size(Cort_Scale,2);
%     
%     Cort_Cat = [Cort_Freq Cort_Rate Cort_Scale];
%     C = zeros(Points,3);
    Cort_Cat = Load_Spec_Plus_1D_Turk(iWav,Block); %!%!%!
%     figure(1); clf; imagesc(Cort_Cat{5}); figure(2); clf; imagesc(Cort_Cat{6});
%     figure(1); clf; plot(max(Cort_Cat{5},[],2)); figure(2); clf; plot(max(Cort_Cat{6},[],2));
%     figure(3); clf; plot(max(Cort_Cat{5},[],2)-max(Cort_Cat{6},[],2));
%     1;

    nTime = size(Cort_Cat{1},1);
    Points = floor(nTime / Step - Window / Step);
    C = zeros(Points,length(Cort_Cat));
    Ind = zeros(Points,length(Cort_Cat));
%     nCat = size(Cort_Cat,2);
%     for iCat = 1:nCat
%         Cort_Cat(:,iCat) = (Cort_Cat(:,iCat) - mean(Cort_Cat(:,iCat))) / std(Cort_Cat(:,iCat));
%     end

    for iTime = 10:(Points-1)
        %Cat
        Samp = (iTime)*Step + (1:Window);
        Last = min(Samp) - 1;
        for iFeat = 1:length(Cort_Cat)

            nCat = size(Cort_Cat{iFeat},2);
            if Last < nMem
                Prev = zeros(nCat,nMem);
                Prev(:,(end-Last+1):end) = Cort_Cat{iFeat}(1:Last,:)';
            else
                Prev = Cort_Cat{iFeat}((Last - nMem + 1):Last,:)';
            end
            Mean_Prev = Prev * Mem;
    %         STD_Prev = std(Prev,[],2);
            Stim = Cort_Cat{iFeat}(Samp,:);
    %         Stim(:,1:128) = 20*log10(Stim(:,1:128));
            Mean_Stim = mean(Stim,1)';
            

%         [C(iTime+1,1),Ind(iTime,1)] = max((Mean_Stim - Mean_Prev) ./ STD_Prev); %Maybe unnecessary, can cause some strange spikes
%         [D(iTime+1,1),Ind(iTime,1)] = max((Mean_Stim - Mean_Prev)

%         C(iTime+1,1) = max((Mean_Stim(1:128) - Mean_Prev(1:128))./STD_Prev(1:128));
%         C(iTime+1,2) = max((Mean_Stim(129:146) - Mean_Prev(129:146))./STD_Prev(129:146));
%         C(iTime+1,3) = max((Mean_Stim(147:end) - Mean_Prev(147:end))./STD_Prev(147:end));

%         C(iTime+1,1) = max((Mean_Stim(1:128) - Mean_Prev(1:128)));
%         C(iTime+1,2) = max((Mean_Stim(129:146) - Mean_Prev(129:146)));
%         C(iTime+1,3) = max((Mean_Stim(147:end) - Mean_Prev(147:end)));

%         C(iTime+1,1) = max((Mean_Stim(1:128) ./ Mean_Prev(1:128)) - 1); %this would require cutting off more of the start, more affected by low prev
%         C(iTime+1,2) = max((Mean_Stim(129:146) ./ Mean_Prev(129:146)) - 1);
%         C(iTime+1,3) = max((Mean_Stim(147:end) ./ Mean_Prev(147:end)) - 1);

%         C(iTime+1,1) = max((Mean_Stim(1:128))); %this would require cutting off more of the start, more affected by low prev
%         C(iTime+1,2) = max((Mean_Stim(129:146)));
%         C(iTime+1,3) = max((Mean_Stim(147:157)));
%         C(iTime+1,4) = max((Mean_Stim(158:285)));
%         C(iTime+1,5) = max((Mean_Stim(286:end)));

%One line change... I guess
%             [C(iTime+1,iFeat),Ind(iTime+1,iFeat)] = max(Mean_Stim);
            C(iTime+1,iFeat) = mean(Mean_Stim); %may need abs?  or is it znormed
            Ind(iTime+1,iFeat) = 1;
        end
        for iFeat = 2:3
            Stim = Cort_Cat{iFeat}(Samp,:);
            
            Max_Stim = max(Stim,[],1)'; %Max max might be ok I guess
            C(iTime+1,iFeat-1+length(Cort_Cat)) = max(Max_Stim); % %Try max or mean in the window, I guess it shouldn't matter too much
            Ind(iTime+1,iFeat-1+length(Cort_Cat)) = 1;
            %11 and 12
%             figure(1)
%             clf
%             imagesc(1:size(Cort_Cat{iFeat},1),RV,Cort_Cat{iFeat}');
            1;
            Cent_Stim = sum( Stim.^2 * diag(Vec{iFeat}),2) ./ sum(Stim.^2,2);
            C(iTime+1,iFeat-1+length(Cort_Cat)+2) = mean(Cent_Stim); % %Try max or mean in the window, I guess it shouldn't matter too much
            Ind(iTime+1,iFeat-1+length(Cort_Cat)+2) = 1;
            %13 and 14
            
            
        end
        
        iFeat = 2;
        Stim = Cort_Cat{iFeat}(Samp,:);
        
        Cent_Stim = sum( Stim.^2 * diag(abs(Vec{iFeat})),2) ./ sum(Stim.^2,2); %Only abs counts, rather than pos/neg.  Probably pos/neg don't matter anyway
        %Eh, this only goes for rate, scale shouldn't be an issue, no
        %negs.  So moving to own section
        C(iTime+1,iFeat-1+length(Cort_Cat)+4) = mean(Cent_Stim); % %Try max or mean in the window, I guess it shouldn't matter too much
        Ind(iTime+1,iFeat-1+length(Cort_Cat)+4) = 1;
            

    end
%     (22050/176/Step)

%     [b_temp,a_temp] = butter(5,0.2,'low');
%     C = filter(b_temp,a_temp,C);
%     for iFeat = 1:length(Cort_Cat)
%         C(C(:,iFeat) < (mean(C(:,iFeat)) + 2*std(C(:,iFeat))),iFeat) = 0;
%     end
%     figure(1); clf; plot(C(:,5)); figure(2); clf; plot(C(:,6));

    Freq{iWav} = Ind;
    
    M = mean(C,1);
    S = std(C,[],1);
    C = (C - repmat(M,[size(C,1) 1])) ./ repmat(S,[size(C,1) 1]); 
    
    Output{iWav} = C';
    [~,Ind] = max(C,[],2);
    Output2{1,iWav} = Ind;
    Counts = [sum(Ind == 1) sum(Ind == 2) sum(Ind == 3) sum(Ind == 4)];
    Output2{2,iWav} = Counts;
    

    1;
%     figure(55 + iWav)
%     clf
% %     plot((1:Points)*8 + fs,C(:,1)/max(C(:,1)),'k')
%     hold on
%     plot((1:Points)*8 + fs,D(:,1)/max(D(:,1)),'k')
%     plot(Resp{iWav},'r')
% 
%     E = Moving_Average(D,32);
%     figure(155 + iWav)
%     clf
% %     plot((1:Points)*8 + fs,C(:,1)/max(C(:,1)),'k')
%     hold on
%     plot((1:Points)*8 + fs,E(:,1)/max(E(:,1)),'k')
%     plot(Resp{iWav},'r')
%     1;
end
% Freq2 = cell2mat(Freq');
% figure(1)
% clf
% plot(Freq2(:,4))
% 
% figure(2)
% clf
% F_Hist = histc(Freq2(:,4),.5:(max(Freq2(:,4))+.5));
% bar(F_Hist);
save(MatFile,'Output','Output2','Freq')
