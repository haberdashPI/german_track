function result = map_trf(fn,varargin)
  result = [];
  names = model_names();
  for k = 1:length(names)
    n = names{k};
    vals = {};
    for i = 1:length(varargin)
      if ismember('trf',fieldnames(varargin{i}))
        vals{i} = varargin{i}.trf.(n);
      else
        vals{i} = varargin{i}.(n);
      end
    end

    result.(n) = fn(vals{:});
  end
end
