
# TODO: once this is basically working, convert to an all julia implementation:
# I don't really need the telluride toolbox to do what I'm doing right now and
# I can optimize it better if it's all in julia

function trf_train(prefix,args...;kwds...)
    cachefn(@sprintf("%s_avg",prefix),trf_train,args...;kwds...)
end

function trf_train(prefix,)

function trf_train(prefix,eeg::MatEEGData,stim_info,lags,indices,stim_fn;name="Training")
    sum_model = Float64[]

    @showprogress name for i in indices
        stim = stim_fn(i)
        if size(stim,2) > 1
            stim = sum(stim,dims=2)
        end
        # TODO: put this inside find trf
        stim_envelope = find_envelope(stim,samplerate(eeg))
        mat"response = $(eeg.data).trial{$i};"

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
    N = round(Int,size(stim,1)/samplerate(stim)*tofs)
    result = zeros(N)
    window_size = 1.5/tofs
    energy = stim.^2
    toindex(t) = clamp(round(Int,t*samplerate(stim)),1,size(stim,1))

    for i in 1:N
        t = i/tofs
        from = toindex(t-window_size)
        to = toindex(t+window_size)
        result[i] = mean(view(energy,from:to,:))
    end

    result
end

function old_find_envelope(stim,tofs)
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
        if i <= 0 || i > size(x,1)
            padded[columns .+ ncol*(ii-1)] .= 0
        else
            padded[columns .+ ncol*(ii-1)] .= x[i,:]
        end
    end

    padded
end

# TODO: use the debug function to debug this optimized lagouter function

function withlags(x,lags)
    y = similar(x,size(x,1),size(x,2)*length(lags))
    for r in axes(x,1)
        for (l,lag) in enumerate(lags)
            for c in axes(x,2)
                r_ = r + lag
                if r_ <= 0
                    y[r,(l-1)*size(x,2)+c] = 0
                elseif r_ > size(x,1)
                    y[r,(l-1)*size(x,2)+c] = 0
                else
                    y[r,(l-1)*size(x,2)+c] = x[r_,c]
                end
            end
        end
    end

    y
end


function find_trf(stimulus,response,dir,lags;K=0.2)
    # L = length(lags)
    # N = size(response,2)*L
    # M = size(envelope,2)
    xlag = withlags(stimulus,lags)
    XX = Symmetric(BLAS.syrk('U','N',1.0,xlag),:U)
    XX = (1-K).*XX .+ I*K*mean(eigvals(XX))
    XY = envelope'xlag
    result = XX\XY'
    # [(feat x lag) x (feat x lag)] * [(featre x lag) x channels] = [(feat x lag) x channels]
    # XX*result = XY'

    result = reshape(result,length(lags),size(stimulus,2),size(response,2))
    permutedims!(result,(1,3,2))
end

function find_trf(stimulus,response::MxArray,dir,lags)
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

