export withlags, decode_test_cv, decoder, RegNorm

using MetaArrays
using Printf
using DataFrames
using StatsBase
using Statistics
using CorticalSpectralTemporalResponses
using DSP
using ProximalAlgorithms
using ProximalOperators
using LambdaFn

using Infiltrator

################################################################################
# testing and training

function decoder(method;prefix,group_suffix="",name="Training",K,
    sources,progress=Progress(K*length(sources),1,desc=name),
    kwds...)

    result = cachefn(@sprintf("%s_avg%s",prefix,group_suffix),
        decoder_,method;prefix=prefix,name=name,progress=progress,
        sources=sources,K=K,
        __oncache__ = () ->
            progress_update!(progress,K*length(sources)),
        kwds...)
end

function decoder_(method;K,prefix,eeg,lags,indices,stim_fn,name="Training",
        sources,bounds=all_indices,progress,kwds...)

    models = [[Array{Float64}(undef,0,0,0) for k in 1:K]
        for i in 1:length(sources)]

    for (k,(train,_)) in enumerate(folds(K,indices))
        for (source_index,source) in enumerate(sources)

            filename = @sprintf("%s_%s_fold%02d",source,prefix,k)
            stim, response = setup_stim_response(stim_fn, source_index, eeg,
                train, bounds)
            model = cachefn(filename, decoder_helper, method, stim,
                response, lags)
            models[source_index][k] = model

            progress_update!(progress)
        end
    end

    models
end

function withlags(x,lags)
    if lags == 0:0
        x
    end

    nl = length(lags)
    n,m = size(x)
    y = similar(x,size(x,1),m*nl)
    z = zero(eltype(y))
    for I in CartesianIndices(x)
        for (l,lag) in enumerate(lags)
            r,c = I[1],I[2]
            r_ = r - lag
            y[r,(l-1)*m+c] = 0 < r_ <= n ? x[r_,c] : z
        end
    end
    y
end
function min_length(stim,eeg,i)
    response = eegtrial(eeg,i)
    min(size(stim,1),trunc(Int,size(response,2)))
end

function setup_stim_response(stim_fn,source_i,eeg,indices,bounds)

    # okay.. the reason this is a problem is because the joint_other source
    # generates a random source, and *whicH* source it generates randomlly
    # various from trial to trial, just need to fix that, and then it should
    # work just fine

    stim_result = mapreduce(vcat,indices) do i
        stim, = stim_fn(i,source_i)
        minlen = min_length(stim,eeg,i)
        select_bounds(stim,bounds[i],minlen,samplerate(eeg),1)
    end
    response_result = mapreduce(hcat,indices) do i
        stim, = stim_fn(i,source_i)
        minlen = min_length(stim,eeg,i)
        select_bounds(eegtrial(eeg,i),bounds[i],minlen,samplerate(eeg),2)
    end
    stim_result, response_result'
end

# TODO: we could probably make things even faster if we created the memory XX
# and XY once.

safezscore(x) = std(x) != 0 ? zscore(x) : x
scale(x) = mapslices(safezscore,x,dims=1)
# adds v to the diagonal of matrix (or tensor) x
adddiag!(x,v) = x[CartesianIndex.(axes(x)...)] .+= v
function decoder_helper(l2::NormL2,stim,response,lags)
    X = withlags(scale(response),.-reverse(lags))
    Y = scale(stim)

    k = l2.lambda
    XX = X'X; XY = Y'X
    λ̄ = tr(XX)/size(X,2)
    XX .*= (1-k); adddiag!(XX,k*λ̄)
    result = XX\XY'
    result = reshape(result,size(response,2),length(lags),:)

    result
end

function decoder_helper(reg::ProximableFunction,stim,response,lags)

    X = withlags(scale(response),.-reverse(lags))
    Y = view(scale(stim),:,:)

    solver = ProximalAlgorithms.ForwardBackward(fast=true,adaptive=true,
        verbose=true, maxit=20000,tol=1e-3)
    _, A, Y, X = code_init(Val(false),Y,X)
    state = Objective(Y,X,A)
    update!(state,Y,X,0.0)
    solver(state.θ,f=state,g=reg)
    result = reshape(state.θ,size(response,1),length(lags),:)

    result
end

decode(response::AbstractArray,model,lags) =
    withlags(scale(response),.-reverse(lags)) * reshape(model,:,size(model,3))

function decode_test_cv(train_method,test_method;prefix,indices,group_suffix="",
    name="Training",sources,K,
    return_models = false,
    progress=Progress(length(indices)*length(sources),1,desc=name),
    train_prefix,kwds...)

    results = cachefn(@sprintf("%s_for_%s_test%s",prefix,train_prefix,group_suffix),
        decode_test_cv_,train_method,test_method;train_prefix=train_prefix,
        prefix=prefix,K=K,
        indices=indices,progress=progress,sources=sources,
        __oncache__ = () ->
            progress_update!(progress,K*length(sources)),
        kwds...)

    results
end

function single(x::Array)
    @assert(length(x) == 1)
    first(x)
end
single(x::Number) = x
apply_method(::typeof(cor),pred,stim) = (value = single(cor(vec(pred),vec(stim))),)
apply_method(fn,pred,stim) = fn(pred,stim)

using Infiltrator

function decode_test_cv_(method,test_method;prefix,eeg,model,lags,indices,stim_fn,
    bounds=all_indices,sources,train_source_indices,progress,train_prefix, K)

    df = DataFrame()
    models = DataFrame()

    for (k,(train,test)) in enumerate(folds(K,indices))
        for (source_index, source) in enumerate(sources)
            train_source = sources[train_source_indices[source_index]]
            model = loadcache(@sprintf("%s_%s_fold%02d",train_source,train_prefix,k))

            for i in test
                stim,response = setup_stim_response(stim_fn, source_index, eeg,
                    [i], bounds)
                _, stim_id = stim_fn(i,source_index)

                pred = decode(response,model,lags)

                push!(df,(apply_method(test_method,pred,stim)...,
                    stim = stim, pred = pred,
                    source = source, index = i, stim_id = stim_id))
            end
            push!(models,(model = model, source = source, k = k))

            next!(progress)
        end
    end

    categorical!(df,:source)
    categorical!(models,:source)
    df, models
end
