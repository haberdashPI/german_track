export withlags, testdecode, decoder

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
using JuMP, Convex, COSMO, SCS
using MathOptInterface: OPTIMAL

################################################################################
# testing and training

function decoder(stim_response_for,method;
    prefix,group_suffix="",indices,
    progress=Progress(length(indices),1,desc="Training"),
    kwds...)

    result = cachefn(@sprintf("%s_avg%s",prefix,group_suffix),
        decoder_,stim_response_for,method;prefix=prefix,
        progress=progress,indices=indices,
        __oncache__ = () ->
            progress_update!(progress,length(indices)),
        kwds...)
end

struct QuadN2
    reg::Float64
end
function decoder_(stim_response_for,method::QuadN2,
    prefix,lags,indices,progress,kwds...)

    stim_responses = [stim_response_for(i) for i in indices]
    stim = mapreduce(@λ(_[1]'),hcat,stim_responses)
    response = mapreduce(@λ(_[2]'),hcat,stim_responses)

    regress(stim,response,method.reg)
end

struct CvNorm
    λ::Float64
    norm::Int
end

regularize(A,reg::CvNorm) = reg.λ*norm(vec(A),reg.norm)
function regressSS(x,y,v,tt,reg)
    M,_ = size(x[1])
    H,K,_ = size(y[1])
    @assert length(x) == length(y) "Stimulus and response must have same trial count."
    T = length(x)

    # decoding coefficients
    A = Variable(K,M)

    # mixture weights
    u = [Variable(H) for _ in 1:T]

    # fix values for known weights
    for (i,t) in enumerate(tt); fix!(u[t],v[i,:]); end

    # solve the problem
    trials = (sumsquares(A*x[t] - sum((u[t][h]*y[t][h,:,:]) for h in 1:H))
        for t in 1:T)
    objective = sum(trials) + regularize(A,reg)
    constraints = ([u[t] > 0, u[t] < 1, sum(u[t]) == 1] for t in 1:T)
    problem = minimize(objective,reduce(vcat,constraints))
    solve!(problem, COSMO.Optimizer())

    # return result
    problem.status == OPTIMAL ||
        @warn("Failed to find a solution to problem:\n $problem")

    A.value, reduce(hcat,getproperty.(u,:value))
end

struct SemiSupervised{I}
    labels::Dict{I,Int}
end
function decoder_(stim_response_for,method::SemiSupervised;
    prefix,lags,indices,progress,kwds...)

    # concatenate responses
    # THOUGHT: do we need to concatenate?
    stim_responses = (stim_response_for(i) for i in indices)
    labels = [get(method.labels,i,missing) for i in indices]
    last_label_indices = cumsum(size(s,1) for (s,_) in stim_responses)
    stim = mapreduce(@λ(_[1]),vcat,stim_responses)
    response = mapreduce(@λ(_[2]),vcat,stim_responses)

    model = Model()
    @variable(model,0 ≤ weights[1:T,1:h] ≤ 1)
    @variable(model,X[coef_indices])

    for i in indices
        stim,response = stim_response_for(i)
        n,m = size(stim)
        n_,k = size(response)

    end
    @variable(model,x[stim_indices])
end

cleanstring(i::Int) = @sprintf("%03d",i)
function decoder_(stim_response_for,method::ProximableFunction;
    prefix,lags,indices,progress,kwds...)

    models = Vector{Array{Float64,3}}(undef,length(indices))

    for (j,i) in enumerate(indices)
        filename = string(prefix,"_",cleanstring(i))
        # @warn "Change this code back!!" maxlog=1
        # stim, response = stim_response_for(train ∪ test)
        stim, response = stim_response_for(i)
        models[j] = cachefn(filename, decoder_helper, method, stim,
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

safezscore(x) = std(x) != 0 ? zscore(x) : x
scale(x) = mapslices(safezscore,x,dims=1)
# adds v to the diagonal of matrix (or tensor) x
adddiag!(x,v) = x[CartesianIndex.(Base.axes(x)...)] .+= v
function decoder_helper(l2::NormL2,stim,response,lags)
    X = withlags(scale(response),.-reverse(lags))
    Y = scale(stim)

    k = l2.lambda
    XX = X'X; XY = Y'X
    λ_bar = tr(XX)/size(X,2)
    XX .*= (1-k); adddiag!(XX,k*λ_bar)
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

function testdecode(stim_response_for,train_method,test_method;prefix,
    indices,group_suffix="",
    name="Training",
    progress=Progress(length(indices),1,desc=name),
    train_prefix,kwds...)

    str = @sprintf("%s_for_%s_test%s",prefix,train_prefix,group_suffix)
    results = cachefn(str,
        testdecode_,stim_response_for,train_method,test_method;
        train_prefix=train_prefix,
        prefix=prefix,
        indices=indices,progress=progress,
        __oncache__ = () ->
            progress_update!(progress,length(indices)),
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

function testdecode_(stim_response_for,method::ProximableFunction,test_method;
    prefix,lags,indices,train_indices,progress,train_prefix)

    df = DataFrame()
    models = map(train_indices) do i
        loadcache(string(train_prefix,"_",cleanstring(i)))
    end
    mean_model = mean(models)
    n = length(models)

    for i in indices
        # construct a model from everything but the model for the trial to be
        # tested

        j = findfirst(==(i),train_indices)
        model = if !isnothing(j)
            mean_model*n/(n-1) - models[j]/(n-1)
        else
            mean_model
        end

        stim,response,stim_id = stim_response_for(i)
        pred = decode(response,model,lags)

        push!(df,(apply_method(test_method,pred,stim)...,
            model = model,
            stim = stim, pred = pred,
            index = i, stim_id = stim_id))

        progress_update!(progress)
    end

    df
end
