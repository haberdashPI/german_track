function z = safeadd(x,y)
  if any(isnan(y))
    z = x;
  else
    z = x + y;
  end
end
