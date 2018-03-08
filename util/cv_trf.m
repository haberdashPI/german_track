function trf = cv_trf(grand,instance,n)
  if isempty(instance)
    trf = grand / n;
  else
    trf = (grand - instance)/(n-1);
  end
end
