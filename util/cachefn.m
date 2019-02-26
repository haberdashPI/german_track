function result = cachefn(file,fn,varargin)
%cachefn - cache function result as *.mat file
%
% Syntax: result = cachefn(file,fn,args...)
%
% If the specified *.mat file doesn't exist, call fn with the specified
% arguments. If it does exist, load the results from the file.
  if exist(file,'file')
    content = load(file);
    result = content.result;
  else
    result = fn(varargin{:});
    save(file,'result');
  end
end
