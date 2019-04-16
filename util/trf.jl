
# TODO: once this is basically working, convert to an all julia implementation:
# I don't really need the telluride toolbox to do what I'm doing right now and
# I can optimize it better if it's all in julia

function trf_train(prefix,args...;group_suffix="",kwds...)
    cachefn(@sprintf("%s_avg%s",prefix,group_suffix),
        trf_train_,prefix,args...;kwds...)
end

function trf_train_(prefix,eeg,stim_info,lags,indices,stim_fn;name="Training",
        bounds=nothing)
    sum_model = Float64[]

    @showprogress name for i in indices
        stim = stim_fn(i)
        # for now, make signal monaural
        if size(stim,2) > 1
            stim = sum(stim,dims=2)
        end

        model = cachefn(@sprintf("%s_%02d",prefix,i),find_trf,stim,eeg,i,
            -1,lags,"Shrinkage",isnothing(bounds) ? nothing : bounds[i])
        # model = find_trf(stim_envelope,response,-1,lags,"Shrinkage")
        if isempty(sum_model)
            sum_model = model
        else
            sum_model .+= model
        end
    end

    sum_model
end

# tried to make things faster by computing envelope in julia (to avoid copying
# the entire wav file): doesn't seem to matter much
function find_envelope(stim,tofs)
    N = round(Int,size(stim,1)/samplerate(stim)*tofs)
    result = zeros(N)
    window_size = 1.5/tofs
    toindex(t) = clamp(round(Int,t*samplerate(stim)),1,size(stim,1))

    for i in 1:N
        t = i/tofs
        from = toindex(t-window_size)
        to = toindex(t+window_size)
        result[i] = mean(x^2 for x in view(stim.data,from:to,:))
    end

    result
end

# function find_envelope(stim,tofs)
#     mat "result = CreateLoudnessFeature($(stim.data),$(samplerate(stim)),$tofs)"
#     get_mvariable(:result)
# end

function find_signals(stim,eeg,i,bounds=nothing)
    # envelope and neural response
    fs = mat"$eeg.fsample"
    stim_envelope = find_envelope(stim,fs)
    mat" response = $eeg.trial{$i}; "

    # find shared length
    rsize = mat" size(response,2) "
    min_len = min(size(stim_envelope,1),trunc(Int,rsize));
    if isnothing(bounds)
        start,stop = 10,min_len
    else
        start, stop = clamp.(round.(Int,bounds.*fs), 1, min_len)
    end

    # trim the the signals
    mat" response = response(:,$start:$stop); "
    response = get_mvariable(:response)
    stim_envelope = stim_envelope[start:stop]

    stim_envelope,response
end

function find_trf(stim,eeg,i,dir,lags,method,bounds=nothing)
    stim_envelope,response = find_signals(stim,eeg,i,bounds)
    lags = collect(lags)
    mat"$result = FindTRF($stim_envelope,$response',-1,[],[],($lags)',$method)"
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
        stim_envelope, response = find_signals(stim,eeg,i)

        pred = predict_trf(-1,response,model,lags,"Shrinkage")
        result[j] = cor(pred,stim_envelope)
    end
    result
end

function trf_corr_cv(prefix,args...;group_suffix="",kwds...)
    cachefn(@sprintf("%s_corr%s",prefix,group_suffix),
        trf_corr_cv_,prefix,args...;kwds...)
end

function trf_corr_cv_(prefix,eeg,stim_info,model,lags,indices,stim_fn;name="Testing")
    result = zeros(length(indices))

    @showprogress name for (j,i) in enumerate(indices)
        stim = stim_fn(i)
        if size(stim,2) > 1
            stim = sum(stim,dims=2)
        end
        stim_envelope,response = find_signals(stim,eeg,i)

        subj_model_file = joinpath(cache_dir,@sprintf("%s_%02d.jld2",prefix,i))
        # subj_model = load(subj_model_file,"contents")
        subj_model = cachefn(subj_model_file,find_trf,stim,eeg,i,-1,lags,"Shrinkage")
        n = length(indices)
        r1, r2 = (n-1)/n, 1/n

        pred = predict_trf(-1,response,(r1.*model .- r2.*subj_model),lags,
            "Shrinkage")
        result[j] = cor(pred,stim_envelope)
    end
    result
end

