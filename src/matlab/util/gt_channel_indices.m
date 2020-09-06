function ixs = gt_channel_indices(selection, channels)
    ixs = [];
    for i = 1:length(selection)
        ixs = [ixs find(strcmp(selection{i}, channels))];
    end
end
