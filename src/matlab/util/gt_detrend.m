function [data,w] = gt_detrend(data,orders,varargin)
    p = inputParser;
    p.addOptional('w',[]);
    p.addOptional('basis','polynomials');
    p.addOptional('threshs',[3 3]);
    p.addOptional('niters',[3 3]);
    p.addOptional('wsizes',{[], []});
    parse(p,varargin{:});

    i = 0;
    w = p.Results.w;
    for ord = orders
        i = i + 1;
        [data,w] = nt_detrend(data,ord,w,...
            p.Results.basis,p.Results.threshs(i),p.Results.niters(i),...
            p.Results.wsizes{i});
    end
end
