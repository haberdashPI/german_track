function comp = gt_ascomponent(eeg,comps,unmix)
    if nargin < 3
        unmix = pinv(comps);
    end
    comp = eeg;
    comp.topo = comps;
    comp.unmixing = unmix;
    comp.topolabel = eeg.label; %cellfun(@(x)sprintf('EOG%02d',x),num2cell(1:size(todss,2)),'UniformOutput',0);
end
