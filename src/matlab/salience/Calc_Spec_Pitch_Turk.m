function Calc_Spec_Pitch_Turk(Block)
if nargin < 1
    Block = 2;
end

loadload
Spec_Dir = ['Block' num2str(Block) '/Spec/'];

nWav = length(dir([Spec_Dir 'Spec*.mat']));
% Spec_Dir = './Normed_Specs/';
% Spec_Dir = './EEG_Stim_Spec/';
% Spec_Dir = './Specs/';
fs = 22050;
CF = cochfil(1:129,log2(fs/16000));
for iWav = 1:nWav;
    load([Spec_Dir 'Spec' num2str(iWav) '.mat'])
    th = exp(mean(log(max(Spec(:),1e-3))));
    [Pitch,Sal] = pitch(log(max(Spec',th)) - log(th), CF(1:(end-1)),'pitlet_templates');
    save([Spec_Dir 'Pitch' num2str(iWav) '.mat'],'Pitch','Sal')
end