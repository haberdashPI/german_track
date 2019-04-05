function trf_train(prefix,eeg,stim,lags,indices,stim_fn)
    sum_model = Float64[]

    @showprogress for i in indices
        stim = stim_fn(i)
        if size(stim,2) > 1
            stim = sum(stim,2)
        end
        eegfs = eeg["fsample"]
        stim_envelope =
            mat"CreateLoudnessFeatures($stim,$(samplerate(stim)),$eegfs)"
        response = eeg["trial"][i]

        min_len = min(size(stim_envelope,1),size(response,2))
        resposne = response[:,1:min_len]'
        stim_envelope = stim_envelope[1:min_len]

        model = cachefn(@sprintf("%s_%02d",prefix,i),find_trf,
            stim_envelope,response,-1,[],[],lags,"Shrinkage")
        if isempty(sum_model)
            sum_model = model
        else
            sum_model .+= model
        end
    end
end

function find_trf(envelope,response,dir,arg1,arg2,lags,method)
    mat"""FindTRF(envelope,response,dir,arg1,arg2,lags,method)"""
end

function trf_corr(eeg,stim,model,lags,indices,stim_fn)
    result = zeros(length(indices))

    eegfs = eeg["fsample"]
    stim_envelope =
        mat"CreateLoudnessFeatures($stim,$(samplerate(stim)),$eegfs)"
    response = eeg["trial"][i]

    min_len = min(size(stim_envelope,1),size(response,2))
    resposne = response[:,1:min_len]'
    stim_envelope = stim_envelope[1:min_len]

    model = find_trf(stim_envelope,response,-1,[],[],lags,"Shrinkage")
