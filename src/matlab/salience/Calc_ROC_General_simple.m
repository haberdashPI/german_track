function [Output] = Calc_ROC_General_simple(P_Events,B_Events, Wav_List, ML)

nSet = 5; 
Bin_Size = 2; 
Overlap = 0.75; 
nWav=length(Wav_List);
nFeat = size(P_Events,1);
% predicted bins which are per features for all the files
P_Bins = cell(1,nWav);
%B_Times = cell(1,nWav);
B_Bins = cell(1, nWav);

for iWav = Wav_List
    Bin_Start = 0:(Bin_Size *(1-Overlap)):ML(iWav);
    Bin_End = Bin_Start + Bin_Size;
    nBins = size(Bin_Start,2);
    
    %Behavioral related events and timestamps of events
    %B_Times{iWav} = [ones(nBins,1)*iWav Bin_Start' Bin_End'];
    %nEvents = size(B_Events{iWav},1);
    B_Bins{1,iWav}=zeros(3, nBins)-10;
    for iEvent = 1:size(B_Events{iWav},1)
            Time = B_Events{iWav}(iEvent,1);
            Height = B_Events{iWav}(iEvent,2);
            Hits = Bin_Start < Time & Bin_End > Time;
            B_Bins{1,iWav}(1, Hits) = max(Height,B_Bins{1,iWav}(1, Hits));
            B_Bins{1,iWav}(2, Hits) = max(1,B_Bins{1,iWav}(2, Hits));
            B_Bins{1,iWav}(3, Hits) = 2;
    end   
  
    % Predicted Events and timestamps
    P_Bins{1, iWav} = -10*ones(nFeat, nBins);
    for iFeat = 1:nFeat
        for iEvent = 1:size(P_Events{iFeat,iWav},1)
            Time = P_Events{iFeat,iWav}(iEvent,1);
            Height = P_Events{iFeat,iWav}(iEvent,2);
            if Height == inf
                1;
            end
            Hits = Bin_Start < Time & Bin_End > Time;
            P_Bins{1,iWav}(iFeat, Hits) = max(Height,P_Bins{1,iWav}(iFeat, Hits));
        end
    end
end

%Mix = 1:nFeat;
Floor = 0;

P_Bins = cell2mat(P_Bins)';
B_Bins = cell2mat(B_Bins)';
P_Bins(P_Bins < Floor) = Floor;
B_Bins(B_Bins(:,1) < Floor, :) = Floor;


lThresh=0:0.005:1;
nThresh = length(lThresh);
Hits_Kept = zeros(nSet,nThresh);
FA_Kept = zeros(nSet,nThresh);
Hits_Total = zeros(nSet,nThresh);
FA_Total = zeros(nSet,nThresh);
Predictions_Total = zeros(nSet,nThresh);

for iSet = 1:nSet
    First = 1 + round((iSet-1)/nSet * size(P_Bins,1));
    Last = round((iSet)/nSet * size(P_Bins,1));
    
    Test_Values = P_Bins(First:Last,:);
    Train_Values = P_Bins; 
    Train_Values(First:Last,:) = [];
    
    Train_Labels = B_Bins(:,2);
    Train_Labels(First:Last) = [];
    Test_Labels = B_Bins(First:Last,2);
    level_labels =B_Bins(First:Last,3);
    W=LDA(Train_Values,Train_Labels);
    
    for iThresh = 1:nThresh
        cThresh = lThresh(iThresh);
        Class = [ones(size(Test_Values,1),1) Test_Values] * W'+[log(cThresh),log(1-cThresh)];
        [~,Class] = max(Class,[],2);
        Class = Class - 1;
        Hits_Kept(iSet,iThresh) = sum(Class & Test_Labels == 1);
        Hits_Kept_high(iSet, iThresh)= sum(Class & Test_Labels == 1 & level_labels ==2);
        Hits_Kept_low(iSet, iThresh)= sum(Class & Test_Labels == 1 & level_labels ==1);
        FA_Kept(iSet,iThresh) = sum(Class & Test_Labels ~= 1);
        Hits_Total(iSet,iThresh) = sum(Test_Labels == 1); 
        FA_Total(iSet,iThresh) = sum(Test_Labels ~= 1);
        Predictions_Total(iSet,iThresh) = sum(Class);
    end
end
Hits = sum(Hits_Kept,1) ./ sum(Hits_Total,1);
FA = sum(FA_Kept,1) ./ sum(FA_Total,1);
Hits_set= Hits_Kept./Hits_Total;
FA_set=FA_Kept./FA_Total;
for k=1:nSet
    AUROC_set(k)=AUROC(FA_set(k,:), Hits_set(k,:));
end
Hits_high=sum(Hits_Kept_high,1) ./ sum(Hits_Total,1);
Hits_low=sum(Hits_Kept_low,1) ./ sum(Hits_Total,1);
Prec = sum(Hits_Kept,1) ./ sum(Predictions_Total,1);

Area = AUROC(FA,Hits);
Output.Hits = Hits;
Output.FA = FA;
Output.Area = Area;
Output.W = W;
Output.Hits_high=Hits_high;
Output.Hits_low=Hits_low;
Output.Prec = Prec;
Output.A_set=AUROC_set;

