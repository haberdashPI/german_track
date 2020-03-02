function Resave_Loud_Bark_Turk(Block)
if nargin < 1
    Block = 3;
end
Top = ['Block' num2str(Block)];
if isempty(dir([Top '/Loud_Bark/']))
    mkdir([Top '/Loud_Bark/']);
end

temp = load([Top '/Loud_Sharp.mat']);
for iWav = 1:20
    Loud_Bark = temp.Loud_Bark{iWav};
    save([Top '/Loud_Bark/Loud_Bark' num2str(iWav) '.mat'],'Loud_Bark')
%     temp2 = load(['Loud_Bark/Loud_Bark' num2str(iWav) '.mat']);
%     1;
    Loud_P = temp.Loud_P{iWav};
    save([Top '/Loud_Bark/Loud_P' num2str(iWav) '.mat'],'Loud_P')
%     figure(1); clf; plot(max(temp.Loud_Bark{iWav},[],2)); figure(2); clf; plot(max(temp.Loud_P{iWav},[],2));
%     figure(3); clf; plot(max(temp.Loud_Bark{iWav},[],2)-max(temp.Loud_P{iWav},[],2));
    1;
end