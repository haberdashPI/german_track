export withlags, testdecode, decoder, withlags, regressSS, regressSS2, onehot, CvNorm, decode, tosimplex, regressSS2

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
using StatsFuns
using Underscores
using Random
using TensorCast
using CUDA


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
    @assert I == I_ "Must have the same number of neural and stimulus observations"
    @assert N == N_ "Must have same number of neural and stimulus time points"
    @assert H == H_ "Must have same number of stimulus sources and stimulus weights"

    xᵥ = view(x,:,:,vi)
    xᵤ = view(x,:,:,setdiff(1:size(x,3),vi))
    yᵥ = view(y,:,:,:,vi)
    yᵤ = view(y,:,:,:,setdiff(1:size(y,4),vi))

    A = gpu(randn(F,G))
    u = gpu(rand(H-1,I-J))

    SemiDecodeTrainer(A,xᵤ,yᵤ,u,xᵥ,yᵥ,gpu(v),vi)
end

function loss(A,x,y,w)
    error = zero(eltype(A))
    for i in 1:size(x)[end]
        xi, yi, wi = view(x,:,:,i), view(y,:,:,:,i), view(w,:,i)
        xA = xi*A
        yw = reshape(reshape(yi,:,size(yi,3))*wi,size(yi)[1:2])
        diff = (xA .- yw).^2
        error += sum(diff)

        # NOTE: potentially better than freeing these intermediate computations
        # from the GPU would be doing in place operations but then i'd have to
        # manually write the adjoint... worth considering if the model is too slow
        # as is (which I don't think it is)

        unsafe_gpu_free!(xA)
        unsafe_gpu_free!(yw)
        unsafe_gpu_free!(diff)
    end
    error
end

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
    uis = 1:size(t.u,2)
    vis = 1:size(t.v,2)
    Dᵤ = Flux.Data.DataLoader(t.xᵤ,t.yᵤ,uis,batchsize=batchsize,shuffle=true)
    Dᵥ = Flux.Data.DataLoader(t.xᵥ,t.yᵥ,vis,batchsize=batchsize,shuffle=true)

    for (source,(_x,_y,wi)) in randmix(Dᵥ,Dᵤ,report_source=true)
        known_weights = source == 1
        _x = gpu(_x)
        _y = gpu(_y)

        if known_weights
            v = t.v[:,wi]
            Δ = Flux.gradient(t.A) do A
                loss(A,_x,_y,v) + reg(vec(A))
            end
            Flux.update!(optimizer,t.A,Δ[1])
            unsafe_gpu_free!(v)
        else
            u = t.u[:,wi]
            Δ = Flux.gradient(t.A,u) do A,_u
                loss(A,_x,_y,tosimplex(_u)) + reg(vec(A))
            end

            Flux.update!(optimizer,t.A,Δ[1])
            t.u[:,wi] .-= apply_byindex!(optimizer,t.u,(:,wi),Δ[2])
            unsafe_gpu_free!(u)
        end
        unsafe_gpu_free!(_x)
        unsafe_gpu_free!(_y)
    end
end

function weights(t::SemiDecodeTrainer,ArrayType=Array)
    w = ArrayType{eltype(t.v)}(undef,size(t.v,1),size(t.u,2)+size(t.v,2))
    w[:,t.vi] = t.v
    w[:,setdiff(1:end,t.vi)] = tosimplex(t.u)
    w
end
function coefs(t::SemiDecodeTrainer,ArrayType=Array)
    ArrayType(t.A)
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
    testx, testy = gpu.((trainer.xᵥ,trainer.yᵥ))
    function status()
        @info "Training, on epoch $epoch."
        CUDA.memory_status()
        testcb(decoder)
    end
    throt_status = Flux.throttle(status,status_rate)

    while epoch < epochs
        regressSS2_train!(trainer,regularize,batchsize,optimizer)
        throt_status()
        epoch += 1
    end

    trainer
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
