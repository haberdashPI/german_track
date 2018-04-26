function result = reduce_weights(fn,names,init,models)
  result = [];
  for k = 1:length(names)
    n = names{k};

    if nargin == 4
      acc = init;
    else
      models = init;
      if isfield(models{1},'weights')
        acc = models{1}.weights.(n);
      else
        acc = models{1}.(n);
      end
    end

    for i = 2:length(models)
      if isfield(models{i},'weights')
        acc_next = models{i}.weights.(n);
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
