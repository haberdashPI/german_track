export withlags, testdecode, decoder, withlags, regressSS, onehot, CvNorm, decode

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

struct CvNorm
    λ::Float64
    norm::Int
end

regularize(A,reg::CvNorm) = reg.λ*norm(vec(A),reg.norm)
"""
    regressSS(x,y,v,tt,reg)

"Semi-supervised" decoding of x, a multi-channel signal (EEG) to a mixture of
sources y across a number of trials t. The values x and y index the trials
and are of length t. Each entry of x is a multi-channel signal (EEG) of m x n
where m is thee number of channels, n the numbe rof time points. Each entry
of y is a tensor of h x k x n where h is the number of sources, k the number
of features used to represent each source, and n the number of time points
for each source. The features of y may include time lagged values (e.g. by
using `withlags`).

regressSS infers a decoding matrix A across all trials and sources, and a set
of weighting coefficients w. For most trials the weighting coefficients are
unknown, but in some cases, those instances where a subject provides an
appropriate response, are labeled. This is the semi-supervised element of the
problem. The value v sepcies the known weights of size s x n while tt (of size
s) indicates the trial indices that have these known weights.

reg specifies how the problem should be regularized (norm 1 or norm 2)

"""
function regressSS(x,y,v,tt,reg;settings...)
    M,_ = size(x[1])
    H,K,_ = size(y[1])
    T = length(x)

    @assert length(x) == length(y) "Stimulus and response must have same trial count."
    @assert size(v,1) == 0 || H == size(v,2) "Number of sources must match weight dimension"
    @assert length(tt) == size(v,1) "Number of weights must match number of weight indices"

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
    solve!(problem, COSMO.Optimizer(;settings...))

    # return result
    problem.status == OPTIMAL ||
        @warn("Failed to find a solution to problem:\n $problem")

    if isnothing(A.value) || any(isnothing,getproperty.(u,:value))
        (A = nothing, w = nothing)
    else
        (A = A.value, w = reduce(hcat,getproperty.(u,:value)))
    end
end

cleanstring(i::Int) = @sprintf("%03d",i)

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

safezscore(x) = length(x) == 1 || std(x) == 0 ? x : zscore(x)
scale(x) = mapslices(safezscore,x,dims=1)
# adds v to the diagonal of matrix (or tensor) x
adddiag!(x,v) = x[CartesianIndex.(Base.axes(x)...)] .+= v

function decoder(l2::NormL2,stim,response,lags)
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
