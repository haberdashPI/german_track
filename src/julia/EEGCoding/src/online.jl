using ProximalOperators
using ProximalAlgorithms
using Unitful
using SampledSignals
using LinearAlgebra
using ProgressMeter
using Distributions
using Distributed
using SharedArrays

export attention_marker, attention_prob, online_decode

# TODO: preprcoessing, add lag and add intercept

################################################################################
raw"""
    code(y,X,[state];params...)

Solve single step of online encoding or decoding problem. For decoding, y =
speech stimulus, and X = eeg data. For encoding, y = eeg data, X = speech
stimulus. You should pass a single window of data to the function, with rows
representing time slices, and columns representing channels (for eeg data) or
features (for stimulus data).

The coding coefficients are computed according to the following optimization
problem

```math
\underset{\theta}{\mathrm{arg\ min}} \quad \sum_i
    \lambda^{k-i} \left\lVert y_i - X_i\theta \right\rVert^2 +
    \gamma\left\lVert\theta\right\rVert
```

In the above equation, y is the output, X the input, θ the parameters to
be solved for, k the current window and i includes all windows including the
current window.

Returns an `Objective` object (see below).
"""
function code
end # see full defintion below

# defines the objective of optimization
struct Objective{M} <: ProximalOperators.Quadratic
    A::M
    b::M
    θ::M

    function Objective(y::Union{AbstractVector,AbstractMatrix},
                       X::AbstractMatrix,M)

        new{M}(
            M(zeros(size(X,2),size(X,2))),
            M(zeros(size(y,2),size(X,2))),
            M(zeros(size(X,2),size(y,2)))
        )
    end
end
ProximalOperators.fun_dom(f::Objective) = "AbstractArray{Real}"
ProximalOperators.fun_expr(f::Objective) = "x ↦ x'Ax - 2bx"
ProximalOperators.fun_params(f::Objective) = "" # parameters will be too large...

function update!(f::Objective,y,X,λ)
    BLAS.syrk!('U','T',1.0,X,λ,f.A) # f.A = λ*f.A + X'X
    BLAS.gemm!('T','N',1.0,y,X,λ,f.b) # f.b = λ*f.b + y'X
    f
end

function (f::Objective)(θ,Aθ=BLAS.symm('L','U',f.A,θ))
    # sum(θ'Aθ .- 2.0.*f.b*θ)
    sum(BLAS.gemm!('N','N',-2.0,f.b,θ,1.0,θ'Aθ))
end

function ProximalOperators.gradient!(y::AbstractArray,f::Objective,θ::AbstractArray)
    Aθ = BLAS.symm!('L','U',1.0,f.A,θ,0.0,y)
    f_ = f(θ,Aθ)
    y .= 2.0.*(Aθ .- f.b')
    f_
end

function code_init(gpu::Val{false},y,X)
    eltype(y), Array{eltype(y),2}, y, X
end

function code_init(gpu,y,X)
    error("GPU backend not loaded. Call `using CuArrays`.")
end

function code(y,X,state=nothing;gpu=false,λ=(1-1/30),γ=1e-3,kwds...)
    T, A, y, X = code_init(Val(gpu),y,X)
    state = isnothing(state) ? Objective(y,X,A) : state
    params = merge((maxit=1000, tol=1e-3, fast=true, verbose=false),kwds)

    update!(state,y,X,λ)
    ProximalAlgorithms.ForwardBackward(;params...)
    _, result = solver(state.θ,f=state,g=NormL1(T(γ)))
    # _, result = ProximalAlgorithms.forwardbackward(state.θ,f=state,g=NormL1(T(γ));
    #     params...)
    state.θ .= result

    state
end

################################################################################
"""
    attention_marker(eeg,targets...;params)

Computes an attentional marker, via decoding, for each specified target,
using the L1 norm of the online encoding/decoding coefficients.

Encoding version not yet supported.
"""
function attention_marker
end

asframes(x,sr) = floor(Int,asseconds(x)*sr)
asseconds(x) = x
asseconds(x::Quantity) = ustrip(uconvert(s,x))

function attention_marker(eeg,targets...;
    # marker parameters
    samplerate,
    window=250ms,
    lag=400ms,
    estimation_length=5s,
    min_norm=1e-4,
    progress=true,
    save_coefs=false,
    code_params...)

    ntargets = length(targets)
    nlag = asframes(lag,samplerate)

    eeg = withlags(eeg,UnitRange(sort([0,-nlag])...))
    window_len = asframes(window,samplerate)
    nwindows = cld(size(eeg,1)-nlag,window_len)
    λ = 1 - window_len/(asframes(estimation_length,samplerate))

    marker = map(_ -> Array{Float64}(undef,nwindows),targets)
    coefs = save_coefs ?
        Array{Float64}(undef,nwindows,ntargets,size(eeg,2),size(targets[1],2)) :
        nothing

    function select_windows(xs,length,index)
        start = (index-1)*length+1
        stop = min(size.(xs,1)...,index*length)
        map(x -> view(x,start:stop,:), xs)
    end

    states = fill(nothing,ntargets)
    prog = !(progress isa Bool) ? progress :
        progress ? Progress(nwindows) : nothing
    for w in 1:nwindows
        # TODO: decoding might work better if we allow for an intercept
        # at each step
        windows = select_windows((eeg,targets...),window_len,w)
        eegw = windows[1]
        states = map(windows[2:end],states,1:ntargets) do targetw,state,ti
            state = code(targetw,eegw,state;λ=λ,code_params...)
            save_coefs && (coefs[w,ti,:,:] = state.θ)
            marker[ti][w] = max(norm(state.θ,1), min_norm)

            state
        end
        isnothing(prog) || next!(prog)
    end

    save_coefs ? (marker, coefs) : marker
end

################################################################################
"""
    attention_prob(x,y)

Given two attention markers, x and y, use a batch, probabilistic state space
model to compute the logit-probability that x is being attended to, as
opposed to y.
"""
function attention_prob(m1,m2;
    outer_EM_iter = 20,
    inner_EM_iter = 20,
    newton_iter = 10,
    confidence_interval = 0.9,

    # parameters of the inverse-gamma prior on state-space variances
    mean_p = 0.2,
    var_p = 5,
    a₀ = 2+mean_p^2/var_p,
    b₀ = mean_p*(a₀-1),

    # tuned attended and unattended prior distributions based on a similar trial
    α₀ = [5.7111e+02,1.7725e+03],
    β₀ = [34.96324,2.1424e+02],
    μ₀ = [0.88897,0.1619],

    # initialized attended and unattended Log-Normal distribution parameters
    # based on a similar trial
    ρ_d₀ = [16.3344,8.2734],
    μ_d₀ = [0.88897,0.16195])

    n = min(size(m1,1),size(m2,1))

    # Kalman filtering and smoothing variables
    z_k = zeros(n+1); z_k₁ = zeros(n+1); z_K = zeros(n+1)
    σ_k = zeros(n+1); σ_k₁ = zeros(n+1); σ_K = zeros(n+1)
    S = zeros(n)

    ρ_d = ρ_d₀; μ_d = μ_d₀

    # final results
    z = zeros(n)
    η = fill(0.3,n)

    # outer E & M defintions
    P(m,i) = (1/m)*sqrt(ρ_d[i])*exp(-0.5*ρ_d[i]*(log(m)-μ_d[i])^2)
    function outerE(m1,m2,z)
        p = 1/(1+exp(-z))
        P1_a = P(m1,1); P1_u = P(m1,2)
        P2_u = P(m2,2); P2_a = P(m2,1)
        (p*P1_a*P2_u)/(p*P1_a*P2_u+(1-p)*P1_u*P2_a)
    end

    function outerM(E,ma,mb,i)
        μ = ( sum(E.*log.(ma).+(1.0.-E).*log.(mb)) + n*μ₀[i] )/2n
        s = sum( E.*((log.(ma).-μ_d[i]).^2) .+
                 (1.0.-E).*((log.(mb).-μ_d[i]).^2) )
        ρ = (2n*α₀[i])/( s + n*(2*β₀[i]+(μ_d[i]-μ₀[i]).^2) )

        μ,ρ
    end

    # outer EM
    for i in 1:outer_EM_iter
        E = outerE.(m1,m2,z)
        μ_d[1], ρ_d[1] = outerM(E,m1,m2,1)
        μ_d[2], ρ_d[2] = outerM(E,m2,m1,2)

        # inner EM for updating z's and eta's (M-Step)

        for j in 1:inner_EM_iter

            # Filtering
            for k in 2:n+1
                z_k₁[k] = z_k[k-1]
                σ_k₁[k] = σ_k[k-1] + η[k-1]

                # Newton's Algorithm
                for m in 1:newton_iter
                    a = z_k[k] - z_k₁[k] - σ_k₁[k]*(E[k-1] -
                        exp(z_k[k])/(1+exp(z_k[k])))
                    b = 1 + σ_k₁[k]*exp(z_k[k])/((1+exp(z_k[k]))^2)
                    z_k[k] = z_k[k] -  a/b
                end

                σ_k[k] = 1 / (1/σ_k₁[k] + exp(z_k[k])/((1+exp(z_k[k]))^2))
            end

            # Smoothing
            z_K[n+1] = z_k[n+1]
            σ_K[n+1] = σ_K[n+1]

            for k = reverse(1:n)
                S[k] = σ_k[k]/σ_k₁[k+1]
                z_K[k] = z_k[k] + S[k]*(z_K[k+1] - z_k₁[k+1])
                σ_K[k] = σ_k[k] + S[k]^2*(σ_K[k+1] - σ_k₁[k+1])

            end
            z_k[1] = z_K[1]
            σ_k[1] = σ_K[1]

            # update the eta's
            η = ((z_K[2:end].-z_K[1:end-1]).^2 +
                  σ_K[2:end] .+ σ_K[1:end-1] .- 2.0.*σ_K[2:end].*S .+ 2.0.*b₀) /
                (1.0.+2.0.*(a₀.+1.0))
        end

        # update the z's
        z = z_K[2:end]
    end

    scale = cquantile(Normal(),0.5(1-confidence_interval))
    z, z .+ scale.*sqrt.(η), z .- scale.*sqrt.(η)
end

function online_decode(;prefix,indices,group_suffix="",sources,
    progress=Progress(length(indices)*length(sources),1),
    kwds...)

    # @info "Check cache"

    cachefn(@sprintf("%s_probs%s",prefix,group_suffix),
        online_decode_;prefix=prefix,indices=indices,
        progress=progress,sources=sources,
        __oncache__ = function()
                # @info "Updating progress"
                progress_update!(progress,length(indices)*length(sources))
            end,
            kwds...)
end

function online_decode_(;prefix,eeg,lags,indices,stim_fn,sources,progress,
    attention_min_norm=1e-3,bounds=all_indices,params...)
    defaults = (window=250ms,maxit=250,tol=1e-2,progress=false,lag=250ms,
    min_norm=1e-16,estimation_length=10s,γ=2e-3)

    isempty(indices) && (return [])

    mapreduce(vcat,indices) do i
        cur_bounds = bounds[i]
        stimuli = map(source_i -> stim_fn(i,source_i),eachindex(sources))
        n = min(minimum(s->size(s,1),stimuli),size(eegtrial(eeg,i),2))
        bounded_stim = map(s->select_bounds(s,bounds[i],n,samplerate(eeg),1),stimuli)
        response = select_bounds(eegtrial(eeg,i),bounds[i],n,samplerate(eeg),2)

        markers = cachefn(@sprintf("%s_attn_%03d",prefix,i),attention_marker,
            response',
            bounded_stim...;
            samplerate=samplerate(eeg),
            __oncache__ = () -> progress_update!(progress,length(sources)),
            merge(defaults,params.data)...)

        μ = mean(mean.(markers))
        nmarkers = map(x -> x./μ, markers)

        result = map(enumerate(nmarkers)) do (source_i,marker)
            others = nmarkers[setdiff(1:length(sources),source_i)]
            probs = attention_prob(max.(attention_min_norm,marker),
                max.(attention_min_norm,others...))
            progress_update!(progress)

            marker,probs...
        end
        [result]
    end
end
