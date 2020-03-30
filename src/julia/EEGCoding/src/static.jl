export withlags, testdecode, decoder, withlags, regressSS, regressSS2, onehot, CvNorm, decode, tosimplex

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
using Zygote: @adjoint
using Zygote
using StatsFuns
using Underscores
using Random
using TensorCast

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

tosimplex(x::AbstractMatrix) = tosimplex_(x)[1]
function tosimplex_(x::AbstractMatrix)
    N,M = size(x)
    y = similar(x,N+1,M)
    c = similar(x,N,M)
    c[1,:] .= y[1,:] .= x[1,:]
    for i in 2:N
        y[i,:] .= (1 .- c[i-1,:]).*x[i,:]
        c[i,:] .= c[i-1,:] .+ y[i,:]
    end
    y[N+1,:] .= (1 .- c[N,:])

    y, c
end

@adjoint function tosimplex(x::AbstractMatrix)
    N,M = size(x)
    y, c = tosimplex_(x)
    y, function(Δ)
        Δx = similar(x)
        Δx[1,:] .= Δ[1,:]
        for i in 2:N
            Δx[i,:] = Δ[i,:].*(1 .- c[i-1,:])
        end
        (Δx,)
    end
end

# function tosimplex(z::AbstractArray)
#     K = length(z)+1
#     y = Array{eltype(z)}(undef,K)
#     c = y[1] = z[1]
#     for k in 2:(K-1)
#         y[k] = (1-c)z[k]
#         c += y[k]
#     end
#     y[K] = (1-c)
#     y
# end

# function fromsimplex(x)
#     K = length(x)
#     y = Array{eltype(x)}(undef,K-1)
#     c = y[1] = x[1]
#     for k in 2:(K-1)
#         y[k] = x[k]/(1-c)
#         c += x[k]
#     end
#     y
# end

# @adjoint function tosimplex(z::AbstractArray)
#     K = length(z)+1
#     y = zsimplex(z)
#     y, function(Δ)
#         Δx = Array{eltype(z)}(undef,K-1)
#         Δx[1] = Δ[1]
#         c = y[1]
#         for k in 2:(K-1)
#             Δx[k] = Δ[k]*(1 - c)
#             c += y[k]
#         end
#         (Δx,)
#     end
# end

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

struct SemiDecodeTrainer
    A::AbstractArray
    xᵤ::AbstractArray
    yᵤ::AbstractArray
    u::AbstractArray
    xᵥ::AbstractArray
    yᵥ::AbstractArray
    v::AbstractArray
    vi::Vector{Int}
end

function SemiDecodeTrainer(x,y,v,vi)
    N,F,I = size(x)
    N_,G,H,I_ = size(y)
    H_,J = size(v)
    @show vi[1:min(end,10)]
    @assert I == I_ "Must have the same number of neural and stimulus observations"
    @assert N == N_ "Must have same number of neural and stimulus time points"
    @assert H == H_ "Must have same number of stimulus sources and stimulus weights"

    xᵥ = view(x,:,:,vi)
    xᵤ = view(x,:,:,setdiff(1:size(x,3),vi))
    yᵥ = view(y,:,:,:,vi)
    yᵤ = view(y,:,:,:,setdiff(1:size(y,4),vi))

    # TODO: we would store A and u in the GPU
    A = randn(F,G)
    u = rand(H-1,I-J)

    SemiDecodeTrainer(A,xᵤ,yᵤ,u,xᵥ,yᵥ,v,vi)
end

function loss(A,x,y,w)
    error = 0
    for i in 1:size(x)[end]
        xi, yi, wi = view(x,:,:,i), view(y,:,:,:,i), view(w,:,i)
        xA = xi*A
        @matmul yw[n,g] := sum(h) yi[n,g,h]*wi[h]
        error += sum((xA .- yw).^2)
    end
    error
end

#=
function loss(A,x,y,w)
    # predicted source mixture given neural signature
    @ein Ŷ[n,g,i] := A[f,g]*x[n,f,i]
    # observed source mixture given sources and mixture weightings
    # (weightings may be unknown)
    @ein Y[n,g,i] := w[h,i]*y[n,g,h,i]

    mse(Ŷ,Y)
end
=#

# Copied and modified from Flux.jl: src/optimise/optimisers.jl
const ϵ = Flux.Optimise.ϵ
function apply_byindex!(o::AMSGrad, x, ix, Δ)
    η, β = o.eta, o.beta
    mt, vt, v̂t = get!(o.state, x, (fill!(zero(x), ϵ), fill!(zero(x), ϵ), fill!(zero(x), ϵ)))
    mt, vt, v̂t = view(mt,ix...), view(vt,ix...), view(v̂t,ix...)
    @. mt = β[1] * mt + (1 - β[1]) * Δ
    @. vt = β[2] * vt + (1 - β[2]) * Δ ^ 2
    @. v̂t = max(v̂t, vt)
    @. Δ = η * mt / (√v̂t + ϵ)
end

apply_byindex!(o, x, ix, Δ) = error("Unsupported optimizer type: $(typeof(o))")

#### end copy

function regressSS2_train!(t::SemiDecodeTrainer,reg,batchsize,optimizer)
    # TODO: this false true mapping doesn't work
    # i need randmix to track which source the relements come from
    uis = 1:size(t.u,2)
    vis = 1:size(t.v,2)
    Dᵤ = Flux.Data.DataLoader(t.xᵤ,t.yᵤ,uis,batchsize=batchsize,shuffle=true)
    Dᵥ = Flux.Data.DataLoader(t.xᵥ,t.yᵥ,vis,batchsize=batchsize,shuffle=true)

    for (source,(_x,_y,wi)) in randmix(Dᵥ,Dᵤ,report_source=true)
        known_weights = source == 1
        if known_weights
            # TODO: this is where we would use `cu` to convert the data
            Δ = Flux.gradient(t.A) do A
                loss(A,_x,_y,t.v[:,wi]) + reg(vec(A))
            end
            Flux.update!(optimizer,t.A,Δ[1])
        else
            # TODO: this is where we would use `cu` to convert the data
            Δ = Flux.gradient(t.A,t.u[:,wi]) do A,_u
                loss(A,_x,_y,tosimplex(_u)) + reg(vec(A))
            end

            Flux.update!(optimizer,t.A,Δ[1])
            t.u[:,wi] .-= apply_byindex!(optimizer,t.u,(:,wi),Δ[2])
        end
    end
end

function coefs(t::SemiDecodeTrainer)
    w = Array{eltype(t.v)}(undef,size(t.v,1),size(t.u,2)+size(t.v,2))
    w[:,t.vi] = t.v
    w[:,setdiff(1:end,t.vi)] = tosimplex(t.u)

    t.A, w
end


"""

- `x`: An `NxFxI` tensor of neural responses, where N is the number of time
    points, F the number of neural features and I the number of observations.
- `y`: An `NxGxHxI` tensor, where N is the number of time points (as above),
   G the number of stimulus features, H the number of sources and I the number
   of observations (as above).
- `v`: A `HxJ` matrix of known source weightings where J < I.
- `vi`: A `J` length vector of the known observation indices for `v`.

"""
function regressSS2(x,y,v,vi;regularize=x->0.0,batchsize=32,epochs=2,
        status_rate=5,optimizer,testcb = x -> nothing)
    # testy = y[sample(1:length(x),batchsize,replace=false)]
    trainer = SemiDecodeTrainer(x,y,v,vi)

    epoch = 0
    function status()
        # TODO: change this to a select mini-batch, or something
        testloss = loss(trainer.A,trainer.xᵥ,trainer.yᵥ,trainer.v) + regularize(vec(trainer.A))
        @info "Current test loss (epoch $epoch): $(@sprintf("%5.5e",testloss)) "
        testcb(decoder)
    end
    throt_status = Flux.throttle(status,status_rate)

    while epoch < epochs
        regressSS2_train!(trainer,regularize,batchsize,optimizer)
        throt_status()
        epoch += 1
    end

    return coefs(trainer)
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
