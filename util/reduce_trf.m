function result = reduce_trf(fn,names,init,models)
  result = [];
  for k = 1:length(names)
    n = names{k};

    if nargin == 4
      acc = init;
    else
      models = init;
      if ismember('trf',fieldnames(models{1}))
        acc = models{1}.trf.(n);
      else
        acc = models{1}.(n);
      end
    end

    for i = 2:length(models)
      if ismember('trf',fieldnames(models{i}))
        acc_next = models{i}.trf.(n);
      else
        acc_next = models{i}.(n);
      end

      if isempty(acc)
        acc = acc_next;
      elseif ~isempty(acc_next)
        acc = fn(acc,acc_next);
      end
    end

    result.(n) = acc;
  end
end
