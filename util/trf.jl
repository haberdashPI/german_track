function trf_train(prefix,eeg,stim_info,lags,indices,stim_fn;name="Training")
    sum_model = Float64[]

    @showprogress name for i in indices
        stim = stim_fn(i)
        if size(stim,2) > 1
            stim = sum(stim,dims=2)
        end
        # TODO: put this inside find trf
        stim_envelope = find_envelope(stim,mat"$eeg.fsample")
        mat"response = $eeg.trial{$i};"

        mat"""
        min_len = min(size($stim_envelope,1),size(response,2));
        response = response(:,1:min_len);
        $stim_envelope = $stim_envelope(1:min_len);
        """
        response = get_mvariable(:response)

        model = cachefn(@sprintf("%s_%02d",prefix,i),find_trf,
            stim_envelope,response,-1,lags,"Shrinkage")
        # model = find_trf(stim_envelope,response,-1,lags,"Shrinkage")
        if isempty(sum_model)
            sum_model = model
        else
            sum_model .+= model
        end
    end

    sum_model
end

function find_envelope(stim,tofs)
    mat"result = CreateLoudnessFeature($(stim.data),$(samplerate(stim)),$tofs)"
    get_mvariable(:result)
end

function find_trf(envelope,response,dir,lags,method)
    # XX = X'*X

    # envelope = mxarray(envelope)
    # response = mxarray(response)
    lags = collect(lags)
    mat"$result = FindTRF($envelope,$response',-1,[],[],($lags)',$method)"
    result
end

function predict_trf(dir,response,model,lags,method)
    mat"[~,$result] = FindTRF([],[],-1,$response',$model,($lags)',$method)"
    result
end

function trf_corr(eeg,stim_info,model,lags,indices,stim_fn;name="Testing")
    result = zeros(length(indices))

    @showprogress name for (j,i) in enumerate(indices)
        stim = stim_fn(i)
        if size(stim,2) > 1
            stim = sum(stim,dims=2)
        end
        stim_envelope = find_envelope(stim,mat"$eeg.fsample")
        mat"response = $eeg.trial{$i};"

        mat"""
        min_len = min(size($stim_envelope,1),size(response,2));
        response = response(:,1:min_len);
        $stim_envelope = $stim_envelope(1:min_len);
        """
        response = get_mvariable(:response)

        pred = predict_trf(-1,response,model,lags,"Shrinkage")
        result[j] = cor(pred,stim_envelope)
    end
    result
end

