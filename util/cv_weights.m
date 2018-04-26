function weights = cv_weights(grand,instance,n)
  if isempty(instance)
    weights = grand / n;
  else
    weights = (grand - instance)/(n-1);
  end
end
