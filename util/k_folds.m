function folds = k_folds(k,len)
  fold_size = ceil(len / k);
  folds = cell(k,2);
  for fold = 1:k
    trials = (fold-1)*fold_size+1 : min(len,fold*fold_size);
    others = setdiff(1:len,trials);
    folds{fold,1} = trials;
    folds{fold,2} = others;
  end
end
