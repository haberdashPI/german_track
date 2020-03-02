function [Weights] = Calc_LDA_weights(P_Events,B_Events, Wav_List, ML)

Bin_Size = 2;
Overlap = 0.75;
nFeat = size(P_Events,1);
Labels=[];
Features=[];

for iWav = Wav_List
    Bin_Start = 0:(Bin_Size *(1-Overlap)):ML(iWav);
    Bin_End = Bin_Start + Bin_Size;
    nBins = size(Bin_Start,2);
    B_Bins=-10*ones(1, nBins);
    for iEvent = 1:size(B_Events{iWav},1)
        Time = B_Events{iWav}(iEvent,1);
        Hits = Bin_Start < Time & Bin_End > Time;
        B_Bins(1, Hits) = max(1,B_Bins(1, Hits));
    end
    Labels=[Labels,B_Bins];
    % Predicted Events and timestamps
    P_Bins = -10*ones(nFeat, nBins);
    for iFeat = 1:nFeat
        for iEvent = 1:size(P_Events{iFeat,iWav},1)
            Time = P_Events{iFeat,iWav}(iEvent,1);
            Height = P_Events{iFeat,iWav}(iEvent,2);
            if Height == inf
                1;
            end
            Hits = Bin_Start < Time & Bin_End > Time;
            P_Bins(iFeat, Hits) = max(Height,P_Bins(iFeat, Hits));
        end
    end
    Features=[Features,P_Bins];
end
Floor = 0;
Labels(Labels<Floor)=Floor;
Features(Features<Floor)=Floor;
sq_Features=[Features.^2];
Features=[Features];
Weights=LDA(Features',Labels');
end

