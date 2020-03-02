function [Labels] = Load_Labels_Turk(Extra,Cut)

if nargin < 2
    Cut = 0;
end
if nargin < 1
    Extra = cell(1,6);
end

%Extra structure is described in Calc_Sal1_Features

Labels{1} = 'Spec 50-300';
Labels{2} = 'Spec 300-1000';
Labels{3} = 'Spec 1000-3000';
Labels{4} = '|\DeltaSpec 50-300|';
Labels{5} = 'Pitch Saliency';
Labels{6} = '|\DeltaLoud|';
% Labels{3} = 'Ceps 60';
% Labels{4} = 'Ceps 11';
% Labels{5} = '|\DeltaCeps 11 (s)|';
% Labels{6} = '|\DeltaCeps 11 (l)|';
Labels{7} = 'Pitch';
Labels{8} = '|\DeltaPitch|';
Labels{9} = 'Bright';
Labels{10} = '|\DeltaBright|';
% Labels{11} = '|\DeltaBW|';
Labels{11} = 'BW';
Labels{12} = 'Loudness';
Labels{13} = 'Zero Cross';
Labels{14} = 'Rate Low';
Labels{15} = 'Rate High';
Labels{16} = 'Scale Low';
Labels{17} = 'Scale High';
Labels{18} = 'HMax LM';
Labels{19} = 'Loudness (Perc)';
Labels{20} = 'Sharpness';
% Labels{19} = '|\DeltaBW|';

if ~isempty(Extra{1})
    for iFreq = Extra{1}
        Labels{end+1} = ['Freq' num2str(iFreq)];
    end
end
if ~isempty(Extra{2})
    for iRate = Extra{2}(Extra{2} <= 9);
        Labels{end+1} = ['Rate' num2str(iRate)];
    end
    for iScale = Extra{2}(Extra{2} > 9);
        Labels{end+1} = ['Scale' num2str(iScale-9)];
    end
end
if ~isempty(Extra{3})
    for iCeps = Extra{3}
        Labels{end+1} = ['Ceps' num2str(iCeps)];
    end
end
if ~isempty(Extra{4})
    for iMFCC = Extra{4}
        Labels{end+1} = ['MFCC' num2str(iMFCC)];
    end
end
if ~isempty(Extra{5})
    for iRand = 1:Extra{5}(1)
        Labels{end+1} = ['Rand' num2str(iRand)];
    end
end
if ~isempty(Extra{6})
    for iBark = 1:Extra{6}(1)
        Labels{end+1} = ['Bark' num2str(iBark)];
    end
end

if Cut
    Labels(1:20) = [];
end