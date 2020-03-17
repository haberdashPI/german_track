function [wseg,segnorm,segsd] = gt_segment_outliers(eeg,w,thresh)

    %% find outlier segments
    segsize = round(0.25*eeg.hdr.Fs);
    segn = zeros(70,segsize);
    segsum1 = zeros(70,segsize);
    segsum2 = zeros(70,segsize);
    n = 0;
    for j = 1:length(eeg.trial)
        for k = 1:segsize:size(eeg.trial{j},2)
            n = n+1;
            until = min(k+segsize-1,size(eeg.trial{j},2));
            seg = eeg.trial{j}(:,k:until);
            wt = w{j}(:,k:until);
            at = (k:until) - k + 1;
            segn(:,at) = segn(:,at) + wt;
            segsum1(:,at) = segsum1(:,at) + seg.*wt;
            segsum2(:,at) = segsum2(:,at) + seg.*seg.*wt;
        end
    end
    segsd = sqrt((segsum2  - segsum1^2) ./ segn);

    segnorm = zeros(n,1);
    n = 0;
    for j = 1:length(eeg.trial)
        for k = 1:segsize:size(eeg.trial{j},2)
            n = n+1;
            until = min(k+segsize-1,size(eeg.trial{j},2));
            seg = eeg.trial{j}(:,k:until);
            wt = w{j}(:,k:until);
            at = (k:until) - k + 1;
            segnorm(n) = max(mean(abs(seg.*wt ./ segsd(:,at)),1));
        end
    end

    wseg = w;
    n = 0;
    for j = 1:length(eeg.trial)
        for k = 1:segsize:size(eeg.trial{j},2)
            n = n+1;
            until = min(k+segsize-1,size(eeg.trial{j},2));
            if segnorm(n) > thresh
                wseg{j}(:,k:until) = 0;
            end
        end
    end

end
