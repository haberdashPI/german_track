export withlags, testdecode, decoder, withlags, regressSS, regressSS2, onehot, CvNorm, decode, tosimplex, zsimplex

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
using Flux
using Flux: mse, onehot
using TransformVariables
using Zygote: @adjoint
using Zygote
using StatsFuns

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
    solve!(problem, COSMO.Optimizer; settings...)

    # return result
    problem.status == OPTIMAL ||
        @warn("Failed to find a solution to problem:\n $problem")

    if isnothing(A.value) || any(isnothing,getproperty.(u,:value))
        (A = nothing, w = nothing)
    else
        (A = A.value, w = reduce(hcat,getproperty.(u,:value)))
    end
end

struct SemiDecoder{N,T}
    A::Array{T,2}
    u::Array{NTuple{N,T},2}
end
Flux.params(x::SemiDecoder) = Flux.params(x.A,x.u)

tosimplex(x::Tuple{},c) = (1-c,)
function tosimplex(x::NTuple{N,T},c=zero(T)) where {N,T}
    yi = (1-c)x[1]
    (yi, tosimplex(Base.tail(x),c+yi)...)
end
fromsimplex(x::NTuple{1},c) = ()
function fromsimplex(x::NTuple{N,T},c=zero(T)) where {N,T}
    (x[1]/(1-c), fromsimplex(Base.tail(x),c+x[1])...)
end

function tosimplex(z::AbstractArray)
    K = length(z)+1
    y = Array{eltype(z)}(undef,K)
    c = y[1] = z[1]
    for k in 2:(K-1)
        y[k] = (1-c)z[k]
        c += y[k]
    end
    y[K] = (1-c)
    y
end

function fromsimplex(x)
    K = length(x)
    y = Array{eltype(x)}(undef,K-1)
    c = y[1] = x[1]
    for k in 2:(K-1)
        y[k] = x[k]/(1-c)
        c += x[k]
    end
    y
end

@adjoint function tosimplex(z::AbstractArray)
    K = length(z)+1
    y = zsimplex(z)
    y, function(Δ)
        Δx = Array{eltype(z)}(undef,K-1)
        Δx[1] = Δ[1]
        c = y[1]
        for k in 2:(K-1)
            Δx[k] = Δ[k]*(1 - c)
            c += y[k]
        end
        (Δx,)
    end
end

function SemiDecoder(x,y,v,tt)
    N,M,H,I = size(x)
    N,K,I = size(y)
    V = length(tt)

    @assert(size(x,4) == I,
        "Stimulus and response must have same trial count.")
    @assert(size(x,1) == N,
        "Stimulus and response must have number of time points.")
    @assert(V == size(v,1),
        "Number of weights must match number of weight indices")
    @assert(size(v,1) == 0 || H == size(v,2),
        "Number of sources must match weight dimension")

    A = randn(M,K)
    u = mapslices(Tuple,rand(I - V,H-1),dims=2)
    indices = in.(1:length(x),Ref(Set(tt)))
    SemiDecoder(A,u,indices)
end

function loss(model::SemiDecoder{T},x,y,uindex::Int) where T
    u = model.u[uindex]
    w = zsimplex(clamp.(u,zero(T),one(T)))
    mse(vec(model.A*x),vec(sum(w.*y,dims=1)))
end
loss(model::SemiDecoder,x,y,v::Array) = mse(vec(model.A*x),vec(sum(v.*y,dims=1)))

# THERE'S a bug!!! the indices of x and y do not correspond to those of tt
# (one is for the whole data set, the other for the batch)
function loss(model::SemiDecoder,x,y,v,tt)
    T = length(x)
    L = 0.0
    ui = 0
    vi = 0
    for i in 1:T
        if model.indices[i]
            vi += 1
            L += loss(model,x[i],y[i],vec(v[tt[vi],:]))
        else
            ui += 1
            L += loss(model,x[i],y[i],ui)
        end
    end

    # impose cost when u is outside the 0,1 boundary
    for ui in model.u
        dist = abs.(ui - 0.5)
        L += sum(max.(0,dist - 0.5)^2)
    end

    L
end

"""

- `x`: An `NxFxI` tensor of neural responses, where N is the number of time
    points, F the number of neural features and I the number of observations.
- `y`: An `NxGxHxI` tensor, where N is the number of time points (as above),
   G the number of stimulus features, H the number of sources and I the number
   of observations.
- `v`: A `HxJ` matrix of known source weightings where J < I.
- `vi`: A `J` length vector of the known observation indices for `v`.

"""
function regressSS2(x,y,v,vi;regularize=x->0.0,batchsize=100,epochs=2,
        status_rate=5,optimizer,testcb = x -> nothing)
    # testy = y[sample(1:length(x),batchsize,replace=false)]
    testx = x
    testy = y

    N,F,I = size(x)
    N,G,H,I = size(y)
    H,J = size(v)

    xᵥ = view(x,:,:,vi)
    xᵤ = view(x,:,:,setdiff(1:size(x,3),vi))
    yᵥ = view(y,:,:,:,vi)
    yᵤ = view(y,:,:,:,setdiff(1:size(y,4),vi))

    A = randn(F,G)
    u = mapslices(Tuple,rand(H,I-J),dims=1)

    Dᵤ = Flux.Data.DataLoader(xᵤ,yᵤ,u,batchsize=batchsize,shuffle=true)
    Dᵥ = Flux.Data.DataLoader(xᵥ,yᵥ,batchsize=batchsize,shuffle=true)

    function loss(x,y,w)
        # TODO: problem, we don't have w, we have a tuple of u
        # this gives a good conceptualization of the implementation goal, however
        # can we use reinterpret?? should be able to do that using SVector
        @ein Ŷ[n,g,i] := A[f,g]*x[n,f,i]
        @ein Y[n,g,i] := w[h,i]*y[n,g,h,i]
        mse(Ŷ,Y)
    end

    # from here on, we can compute the gradients with respect to x,y,w or x,y
    # depending on whether we are using xᵥ or xᵤ.

    epoch = 0
    function status()
        testloss = loss(decoder,testx,testy,v,tt) + regularize(vec(decoder.A))
        # Δ = Flux.gradient(@λ(loss(decoder,testx,testy,v,tt) + regf(decoder)),Flux.params(decoder))
        # @info "Decoder Gradient $(Δ[decoder.A])"
        # @info "Weight Gradient $(Δ[decoder.u])"
        @info "Current test loss (epoch $epoch): $(@sprintf("%5.5e",testloss)) "
        testcb(decoder)
    end
    throt_status = Flux.throttle(status,status_rate)

    for n in 1:epochs

        epoch = n
        Flux.train!(@λ(loss(decoder,_x,_y,v,tt) + regf(decoder)),
            Flux.params(decoder), data, optimizer, cb = throt_status)
    end
    status()

    T = length(x)
    w = Array{eltype(v)}(undef,length(x),size(v,2))
    w[tt,:] = v
    if length(decoder.u) > 0
        w[setdiff(1:T,tt),:] = mapslices(zsimplex,decoder.u,dims=2)
    end
    decoder.A, w
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
