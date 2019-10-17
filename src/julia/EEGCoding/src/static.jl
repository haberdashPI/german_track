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

################################################################################
# testing and training

function decoder(stim_response_for,method;prefix,group_suffix="",K,
    progress=Progress(K,1,desc="Training"),
    kwds...)

    result = cachefn(@sprintf("%s_avg%s",prefix,group_suffix),
        decoder_,stim_response_for,method;prefix=prefix,
        progress=progress,K=K,
        __oncache__ = () ->
            progress_update!(progress,K),
        kwds...)
end

function decoder_(stim_response_for,method;K,prefix,lags,indices,
    progress,kwds...)

    models = Vector{Array{Float64,3}}(undef,K)

    for (k,(train,_)) in enumerate(folds(K,indices))
        filename = @sprintf("%s_fold%02d",prefix,k)
        stim, response = stim_response_for(train)
        models[k] = cachefn(filename, decoder_helper, method, stim,
            response, lags)

        progress_update!(progress)
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

# a more general, but much slower regularized solver
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

function decode_test_cv(stim_response_for,train_method,test_method;prefix,
    indices,group_suffix="",
    name="Training",K,
    return_models = false,
    progress=Progress(length(indices),1,desc=name),
    train_prefix,kwds...)

    results = cachefn(@sprintf("%s_for_%s_test%s",prefix,train_prefix,group_suffix),
        decode_test_cv_,stim_response_for,train_method,test_method;
        train_prefix=train_prefix,
        prefix=prefix,K=K,
        indices=indices,progress=progress,
        __oncache__ = () ->
            progress_update!(progress,K*length(indices)),
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

function decode_test_cv_(stim_response_for,method,test_method;prefix,eeg,
    lags,indices,sources,train_source_indices,progress,train_prefix,K)

    df = DataFrame()
    models = DataFrame()

    for (k,(train,test)) in enumerate(folds(K,indices))
        for (source_index, source) in enumerate(sources)
            model = loadcache(@sprintf("%s_fold%02d",train_prefix,k))

            for i in test
                stim,response,stim_ids = stim_response_for(i)
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
