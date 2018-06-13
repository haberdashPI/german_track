function result = mean_model(fn,arg1,arg2)
  result = [];
  names = model_names();
  for k = 1:length(names)
    n = names{k};

    if nargin == 2
      models = arg1;

      m = models{1}.(n);
      for i = 2:length(models)
        m = fn(m,models{i}.(n));
      end
    elseif nargin == 3
      m = arg1;
      models = arg2;

      for i = 2:length(models)
        m = fn(m,models{i}.(n));
      end
    else
      error('Unexpected number of arguments');
    end

    result.(n) = m;
  end
end
