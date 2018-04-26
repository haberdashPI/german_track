function result = map_weights(fn,names,varargin)
  result = [];
  for k = 1:length(names)
    n = names{k};
    vals = {};
    c = 0;
    for i = 1:length(varargin)
      if isfield(fieldnames(varargin{i}),'weights')
        val = varargin{i}.weights.(n);
      else
        val = varargin{i}.(n);
      end
      c = c+1;
      vals{c} = val;
    end

    result.(n) = fn(vals{:});
  end
end
