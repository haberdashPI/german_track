
function [len_stim,s1,s2,s3] = equalize_lengths(s1,s2,s3)
    if length(s1) > 1
        len_stim = max(length(s1),length(s2));
        s1 = [s1; zeros(len_stim-length(s1),1)];
        s2 = [s2; zeros(len_stim-length(s2),1)];
        s3 = [s3(1:min(len_stim,length(s3))); zeros(len_stim-length(s3),1)];
    else
        len_stim = max(s1,s2);
    end
end
