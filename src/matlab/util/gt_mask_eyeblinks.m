function [wout,pcas] = gt_mask_eyeblinks(eeg,w,eog,use_comp,thresh)
    eegcat = gt_fortrials(@(x)x,eeg);
    eegcat = vertcat(eegcat{:});
    wcat = vertcat(w{:});

    %% filter eyeblink channels
    time_shifts = 0:10;
    eyes = eegcat(:,eog);
    [B,A]=butter(2,1/(eeg.hdr.Fs/2), 'high');
    tmp = filter(B,A,eyes);
    b = fir1(512,[8/(eeg.hdr.Fs/2) 14/(eeg.hdr.Fs/2)],'stop');
    tmp= filtfilt(b,1,tmp);

    %% compute TSPCAs
    pcas=nt_pca(tmp,time_shifts,4);

    %% compute a mask, to select regions of probable eyeblinks
    c = abs(hilbert(pcas(:,use_comp)));
    mask=abs(c)>thresh*median(abs(c));

    wout = ones(size(wcat));
    wout(mask) = 0;
    wout = gt_asfieldtrip(eeg,wout,'croplast',10).trial;
    wout{end} = [wout{end} zeros(size(wout{end},1),10)];
end
