
# TODO: once this is basically working, convert to an all julia implementation:
# I don't really need the telluride toolbox to do what I'm doing right now and
# I can optimize it better if it's all in julia

function trf_train(prefix,args...;kwds...)
    cachefn(@sprintf("%s_avg",prefix),trf_train,args...;kwds...)
end

function trf_train_(prefix,eeg,stim_info,lags,indices,stim_fn;name="Training")
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

# TODO: WIP think more about how to do this right
# (also is this really goign to help?)
function zero_pad_rows(x::AbstractMatrix,indices::UnitRange)
    columns = axes(x,2)
    ncol = size(x,2)
    padded = similar(x,size(x,2)*length(indices))
    for (ii,i) in enumerate(indices)
        @show collect(columns .+ ncol*(ii-1))
        if i <= 0 || i > size(x,1)
            padded[columns .+ ncol*(ii-1)] .= 0
        else
            padded[columns .+ ncol*(ii-1)] .= x[i,:]
        end
    end

    padded
end

# TODO: use the debug function to debug this optimized lagouter function
function lagouter(x,lags::UnitRange)
    n = length(lags)
    xx = similar(x,size(x,2)*n,size(x,2)*n)

    for r in axes(x,1)
        BLAS.syr!('U',1.0,zero_pad_rows(x,r .+ lags),xx)
    end

    Symmetric(xx,:U)
end

function find_trf(envelope,response,dir,lags,method)
    # L = length(lags)
    # N = size(response,2)*L
    # M = size(envelope,2)

    # XY = Array{Float64,2}(undef,M,M*length(lags))

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



function trf_corr_cv(prefix,eeg,stim_info,model,lags,indices,stim_fn;name="Testing")
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

        subj_model_file = joinpath(cache_dir,@sprintf("%s_%02d.jld2",prefix,i))
        subj_model = load(subj_model_file,"contents")
        n = length(indices)
        r1, r2 = (n-1)/n, 1/n

        pred = predict_trf(-1,response,(r1.*model .- r2.*subj_model),lags,
            "Shrinkage")
        result[j] = cor(pred,stim_envelope)
    end
    result
end
