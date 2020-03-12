function w = gt_cropend_weights(x,n)
    w = ones(size(x,1),1);
    w(1:n) = 0;
    w((end-n+1):end) = 0;
end
