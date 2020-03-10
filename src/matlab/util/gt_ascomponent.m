function comp = gt_ascomponent(eeg,comps)
    comp = eeg;
    comp.topo = comps;
    comp.unmixing = pinv(comps);
    comp.topolabel = eeg.label; %cellfun(@(x)sprintf('EOG%02d',x),num2cell(1:size(todss,2)),'UniformOutput',0);
end
