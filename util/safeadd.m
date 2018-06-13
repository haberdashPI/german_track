function z = safeadd(x,y)
  if isstruct(x)
    z = [];
    names = fieldnames(x);
    for i = 1:length(names)
      n = names{i};
      z.(n) = safeadd(x.(n),y.(n));
    end
  elseif any(isnan(y))
    z = x;
  else
    z = x + y;
  end
end
