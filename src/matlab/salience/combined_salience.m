function [Output] = combined_salience(P_Events, LDA_W, ML)

Bin_Size = 2;
Overlap = 0.75;
nFeat = size(P_Events,1);
Features=[];
Wav_List=1:size(P_Events,2);

Output=cell(size(P_Events,2),1);

for iWav = Wav_List
    Bin_Start = 0:(Bin_Size *(1-Overlap)):ML(iWav);
    Bin_End = Bin_Start + Bin_Size;
    nBins = size(Bin_Start,2);
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
    Features=P_Bins;
    Floor = 0;
    Features(Features<Floor)=Floor;
    LDA_out = LDA_W*[ones(1,size(Features,2));Features];
    Output{iWav}= LDA_out(2,:)-LDA_out(1,:);
end

end
