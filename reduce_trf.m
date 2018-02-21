function result = reduce_trf(fn,models)
  result = [];
  names = model_names();
  for k = 1:length(names)
    n = names{k};

    if ismember('trf',fieldnames(models{1}))
      m = models{1}.trf.(n);
    else
      m = models{1}.(n);
    end

    for i = 2:length(models)
      if ismember('trf',fieldnames(models{i}))
        mn = models{i}.trf.(n);
      else
        mn = models{i}.(n);
      end

      m = fn(m,mn);
    end

    result.(n) = m;
  end
end
