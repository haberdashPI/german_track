function ft = gt_tofieldtrip(matrix,head,channels)
    if nargin == 2
        channels = 1:size(matrix,2);
    end
    ft = [];
    ft.trial = {matrix(:,channels)'};
    ft.label = head.label(channels);
    ft.time = {(1:size(matrix,1))/head.Fs};
    ft.hdr = head;
end
