function [M_Events,M_Diff] = Calc_Feat_Events(Flip, features, sampling_freq, MV, delay)
%Input argument Flip - if Flip == 1, then it looks at decreases in the
%feature rather than increases. Default is look for increases in the
%feature
num_files=length(features);
Wav_List = 1:num_files;
if nargin<4
    MV=1.5;
    delay=0;
elseif nargin<5
    delay=0;
end


% temp = load('Mean_Slope_1D_Plus_Plus_Norm.mat');

M = features;

dM = cell(1,num_files);
for iWav = 1:num_files

    dM{iWav} = zeros(size(M{iWav},1),size(M{iWav},2)-1);
    for iFeat = 1:size(M{iWav},1)
        M_Temp = M{iWav}(iFeat,:);
        temp = diff(M_Temp); %Spec only?
        
        temp = movmean(temp,round(MV*sampling_freq));
        temp = movmean(temp,round(MV*sampling_freq));
        temp = movmean(temp,round(MV*sampling_freq));
        
        dM{iWav}(iFeat,:) = temp;
    end
end

M = dM;
clear dM;


nFeat = size(M{1},1);
M_Events = cell(nFeat,num_files);
M_Diff = cell(nFeat,num_files);
%Ref was 1.5 seconds for calculating events from Merve's model, huh
Ref = 1.5;


%ML = [ones(1,16)*1875 1289 1289 1243 1243]; %max length


parfor iWav = Wav_List
    for iFeat = 1:nFeat
        M_Temp = M{iWav}(iFeat,:);
        dM = M_Temp;

        
        [PM,M_Ind] = findpeaks(dM * Flip * -1,'minpeakdistance',round(Ref*sampling_freq)); %only negate if flip
 %       Cut = M_Ind > ML(iWav);
 %       PM(Cut) = [];
  %      M_Ind(Cut) = [];


        M_Events{iFeat,iWav} = [M_Ind'/sampling_freq-delay PM']; %Frak
       % M_Events{nFeat+1,iWav} = [M_Events{nFeat+1,iWav}; M_Events{iFeat,iWav}(:,1)];
        M_Diff{iFeat,iWav} = dM;

    end

end

