function [y,toGood,fromGood]=gt_interpolate_bad_channels(x,iBad,closest,d,n)
%y=interpolate_bad_channels(x,iBad,coordinates,n) - interpolate bad channels from good
%
%  y: interpolated data
%
%  x: data to interpolate
%  iBad: indices of bad channels
%  coordinates: coordinate map (see nt_proximity)
%  n: number of neighboring channels to use [default: 3]
%
% NoiseTools;
%
% Copied version correct issue with using `nt_proximity`

nt_greetings;

if nargin<4;
    error('!');
end
if nargin<5;
    n=3;
end

nchans=size(x,2);
toGood=eye(nchans);
toGood(:,iBad)=[];

closest = [(1:size(x,2))' closest];
d = [zeros(size(x,2),1) d];
if size(closest,1)~=nchans; error('!'); end

fromGood=eye(nchans);
for iChan=iBad
    iOthers=closest(iChan,:);
    iOthers=setdiff(iOthers, iBad, 'stable'); % don't include bad channels
    if numel(iOthers)<n; error('!'); end
    iOthers=iOthers(1:n);
    w=1./(d(iChan,iOthers) + eps);
    w=w/sum(w);
    fromGood(iOthers,iChan)=w;
end
fromGood(iBad,:)=[];

topo=ones(nchans,1);
topo(iBad)=0;

y= x*(toGood*fromGood);

if nargout==0
    figure(100); clf
    subplot 121; nt_imagescc(fromGood);
    subplot 122; nt_topoplot(coordinates,topo);
end
