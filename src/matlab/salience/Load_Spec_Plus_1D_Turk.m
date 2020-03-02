function [Spec_Plus] = Load_Spec_Plus_1D_Turk(iWav,Block)

%2-D rep of brightness, pitch, etc seems not too good, going back to a
%single dimension for those, just to try with the new(ish) mean calculation

%Want to get the spectrogram plus some array of features to complement it.
%I imagine that Pitch would be a good one, not sure what else.  If not
%something that fits easily into a 2d representation, then what?
% Loudness in bark frequency channels?
% Brightness as zeros and ones?
% Rate and Scale as maps?
% Hm.
% Bandwidth is Merve's last one

% normalization...
% z-score and max? across channels or within channels?
% Hm...
if nargin < 2
    Block = 2;
end
Top = ['Block' num2str(Block)];

if nargin < 1
    iWav = 1;
end

[Features, Labels] = Calc_Sal1_Features_Check_Turk(iWav,Block);
Features{iWav} = [zeros(20,20); Features{iWav}; zeros(20,20)];
%Bright is 9, BW is 11

B = Features{iWav}(:,9); %Brightness can only be within the regular spectrogram limits, so will do same as pitch
B = Moving_Average(B,128);
% B = Moving_Average(B,128);

% any interest in value of brightness?  maybe use bandwidth as the value...
% ones and zeros for now, I suppose.
% May need to smooth brightness first (I'm surprised it moves around so
% much)

Bright = fNorm(B);
% CF = 440 * 2 .^ ((-31:97)/24);
% CF(1) = [];
% Ind = abs(repmat(B,1,length(CF)) - repmat(CF,length(B),1));
% Bright = zeros(size(Ind,1)+40,size(Ind,2));
% [~,Ind] = min(Ind,[],2);
% for iTime = 1:length(B)
%     Bright(iTime+20,Ind(iTime)) = 1;
% end
% Bright = [zeros(20,1);B;zeros(20,1)];


% figure(1)
% clf
% imagesc(Bright')
% plot(B)

Flat = Features{iWav}(:,18); %Currently on Loudness (12) instead of BW (BW is 11)
Flat = Moving_Average(Flat,128);
Flat = fNorm(Flat);

SIR = Features{iWav}(:,4); %Currently on Loudness (12) instead of BW (BW is 11)
SIR = Moving_Average(SIR,128);
SIR = fNorm(SIR);

BW = Features{iWav}(:,11); %Currently on Loudness (12) instead of BW (BW is 11)
BW = Moving_Average(BW,128);
Bandwidth = fNorm(BW);
% figure(1)
% clf
% plot(BW)
% max(BW)
% min(BW)
% 1;


% CF = 440 * 2 .^ ((-31:97)/24);
% CF(1) = [];
% Ind = abs(repmat(BW,1,length(CF)) - repmat(CF,length(BW),1));
% Bandwidth = zeros(size(Ind,1)+40,size(Ind,2));
% [~,Ind] = min(Ind,[],2);
% for iTime = 1:length(BW)
%     Bandwidth(iTime+20,Ind(iTime)) = 1;
% end
% % Bandwidth = [zeros(20,1);BW;zeros(20,1)];
% % figure(1)
% % clf
% % imagesc(Bandwidth')
% 1;



Spec = load([Top '\Spec\Spec' num2str(iWav) '.mat']);
Spec = Spec.Spec;

Cort = load([Top '\Cort\Cort' num2str(iWav) '.mat']);
Cort_Rate = abs(Cort.Cort_Rate);
Cort_Rate(:,1:9) = fliplr(Cort_Rate(:,1:9));
Cort_Scale = abs(Cort.Cort_Scale);



clear Cort;

temp = load([Top '\Spec\Pitch' num2str(iWav) '.mat']);
% temp.Pitch = Moving_Average(temp.Pitch,128);
% temp.Pitch = Moving_Average(temp.Pitch,128);

Sal = Moving_Average(temp.Sal,128); %smoothing done here or in calc_mean?
Pitch = Moving_Average(temp.Pitch,128);

% Pitch = zeros(size(Spec));
% CF = 440 * 2 .^ ((-31:97)/24);
% CF(1) = [];
% Ind = abs(repmat(temp.Pitch,1,length(CF)) - repmat(CF,length(temp.Pitch),1));
% [~,Ind] = min(Ind,[],2);
% for iTime = 1:length(temp.Pitch)
%     Pitch(iTime,Ind(iTime)) = temp.Sal(iTime);
% end

Loud_Bark = load([Top '/Loud_Bark/Loud_Bark' num2str(iWav) '.mat']);
Loud_Bark = Loud_Bark.Loud_Bark;
Loud_Bark = [Loud_Bark; zeros(size(Spec,1) - size(Loud_Bark,1),size(Loud_Bark,2))];
Loud_Bark(isnan(Loud_Bark)) = 0;

Loud_P = load([Top '/Loud_Bark/Loud_P' num2str(iWav) '.mat']);
Loud_P = Loud_P.Loud_P;
Loud_P = [Loud_P; zeros(size(Spec,1) - size(Loud_P,1),size(Loud_P,2))];
Loud_P(isnan(Loud_P)) = 0;

% figure(1); clf; plot(max(Loud_Bark,[],2)); figure(2); clf; plot(max(Loud_P,[],2));
% figure(3); clf; plot(max(Loud_Bark,[],2)-max(Loud_P,[],2));
% 1;
    
% figure(1)
% clf
% imagesc(Pitch')
% Spec = Spec ./ max(max(Spec));
% Cort_Rate = Cort_Rate ./ max(max(Cort_Rate));
% Cort_Scale = Cort_Scale ./ max(max(Cort_Scale));
% Pitch = Pitch ./ max(max(Pitch));
% Loud_Bark = Loud_Bark ./ max(max(Loud_Bark)); %?%?%
% Spec = fNorm(20*log10(Spec));


Spec = fNorm(Spec); %also consider no norming, which may affect other models
Cort_Rate = fNorm(Cort_Rate);
Cort_Scale = fNorm(Cort_Scale);
Pitch = fNorm(Pitch);
Loud_Bark = fNorm(Loud_Bark);

% figure(1)
% clf
% imagesc(Pitch')


% Loud_P = fNorm(Loud_P);

% figure(1); clf; plot(max(Loud_Bark,[],2)); figure(2); clf; plot(max(Loud_P,[],2));
% figure(3); clf; plot(max(Loud_Bark,[],2)-max(Loud_P,[],2));
1;


% Spec_Plus = {Spec, Cort_Rate, Cort_Scale, Pitch, Loud_Bark, Loud_P, Bandwidth};
% Spec_Plus = {Spec, Cort_Rate, Cort_Scale, fNorm(Loud_P), Loud_Bark, Pitch, Bright};
Spec_Plus = {Spec, Cort_Rate, Cort_Scale, Bandwidth, Loud_Bark, Pitch, Bright, Sal, Flat, SIR};

% CF = repmat(CF,
1;


function [Output] = fNorm(Input) %Oops
%Huh, I'm not norming?
%Of course I am
%I wasn't norming for tag NoNorm

% Output = Input;

%Hm
%Well, this no longer covers all our features.  So why don't we do the norm
%later.  Still by wave, but after resampling and whatever other stuff I do

% Output = Input ./ max(max(Input));
% 
% m = mean(reshape(Input,numel(Input),1));
% s = std(reshape(Input,numel(Input),1));
% Output = (Input - m) / s;

Output = Input;


