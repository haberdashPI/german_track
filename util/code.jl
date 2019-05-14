module EEGAttentionMarker

using ProximalOperators
using ProximalAlgorithms
using Unitful

# TODO: preprcoessing, add lag and add intercept

################################################################################
"""
    code(y,X,[state];params...)

Solve online encoding or decoding problem. For decoding y = speech stiμlus,
and X = eeg data. For encoding, y = eeg data, X = speech stiμlus. You should
pass α single window of data to the function, with rows representing time
slices, and columsn representing channels (for eeg data) or features (for
stiμlus data).

The coding coefficients are computed according to the following optimization
problem

argmin θ: Σᵢ λ^(k-i) ||yᵢ - Xᵢθ||² + γ||θ||
where y is the output, X the input, and θ the parameters to be solved for


Returns an internal state and α solution
"""
function code; end # see full defintion below

# defines the objective of optimization
struct Objective <: ProximableFunction
    α::Matrix{Float64}
    b::Matrix{Float64}
    θ::Matrix{Float64}
    function Objective(y::Union{Vector,Matrix},X::Matrix)
        new(
            zeros(size(X,2),size(X,2)),
            zeros(size(y,2),size(X,2)),
            zeros(size(X,2),size(y,2))
        )
    end
end
ProximalOperators.is_convex(::Objective) = true
ProximalOperators.is_smooth(::Objective) = true
ProximalOperators.is_separable(::Objective) = true
ProximalOperators.is_quadratic(::Objective) = true
ProximalOperators.is_strongly_convex(::Objective) = true
ProximalOperators.fun_name(f::Objective) = "General Quadratic"
ProximalOperators.fun_dom(f::Objective) = "AbstractArray{Real}, AbstractArray{Real}"
ProximalOperators.fun_expr(f::Objective) = "x ↦ x'Ax - 2bx"
ProximalOperators.fun_params(f::Objective) = "" # parameters will be too large...

function update!(f::Objective,y,X,λ)
    f.α .*= λ; f.α .+= X'X
    f.b .*= λ; f.b .+= y'X
    f
end

function (f::Objective)(θ)
    α,b = f.α,f.b
    y = θ'α*θ; y -= 2 .* b*θ
    sum(y)
end

function ProximalOperators.gradient!(y::AbstractArray,f::Objective,x::AbstractArray)
    α,b = f.α,f.b
    y .= 2 .* α*x .- 2 .* b'
    f(x)
end

function code(y,X,state=nothing;λ=0.9,γ=1e-3,kwds...)
    state = isnothing(state) ? Objective(y,X) : state
    params = merge((maxit=1000, tol=1e-3, fast=true),kwds)

    update!(state,y,X,λ)
    state.θ = ProximalAlgorithms.FBS(state.θ,fs=state,fq=NormL1(γ);
        params...)

    state
end

asseconds(x) = x
asseconds(x::Quantity) = ustrip(uconvert(s,x))

################################################################################
"""
    marker(eeg,targets...;params)

Computes an attentional marker for each specified target, using the L1 norm
of the online decoding coefficients.
"""
function marker(eeg,targets...;
    # marker parameters
    window=250ms,
    lag=400ms,
    estimation_length=5s,
    min_norm=1e-4,
    code_params...)

    # TODO: this thing about the lag doesn't actually make μch sense ... I
    # think that's for compensating for something we're not doing here

    nt = length(targets)
    nlag = floor(Int,asseconds(lag)*samplerate(eeg))+1
    ntime = size(eeg,1)-L+1
    window_len = floor(Int,asseconds(window)*samplerate(eeg))
    nwindows = div(ntime/window_len)
    λ = 1 - window_len/(asseconds(estimation_length)*samplerate(eeg))

    results = map(_ -> Array{Float64}(undef,nwindows),targets)

    window(x;length,index) = x[(1:length) .+ (index-1)*length,:]

    states = fill(nothing,nt)
    for w in 1:nwindows
        # TODO: decoding might work better if we allow for an intercept
        # at each step
        eegw,targetws... = window.((eeg,targets...),length=window_len,index=w)
        states = map(targetws,states,1:nt) do targetw,state,ti
            state = code(targetw,eegw,state;λ=λ,code_params...)
            results[ti][w] = max(norm(state.θ,1), min_norm)

            state
        end
    end

    results
end

################################################################################

sig(x) = 1/(1+exp(-x))

"""
    attention(x,y)

Given two attention markers, x and y, use α batch, probabilistic state space
module to compute smoothed markers.
"""
function attention(x1,x2;
    outer_EM_batch = 20,
    inner_EM_batch = 20,
    newton_iter = 10,

    # inverse-gamma prior
    μ_p = 0.2,
    σ_p = 5,
    a₀ = 2+μ_p^2/σ_p,
    b₀ = μ_p*(a₀-1),

    # TODO: add parameter comments
    ρ_d₀ = [16.3344,8.2734],
    μ_d₀ = [0.88897,0.16195],
    α₀ = [5.7111e+02;1.7725e+03],
    β₀ = [34.96324;2.1424e+02],
    μ₀ = [0.88897;0.1619])

    n = size(x1,1)

    # Kalman filtering and smoothing variables
    z_k_k = zeros(n+1,1);
    z_k_k_1 = zeros(n+1,1);
    sig_k_k = zeros(n+1,1);
    sig_k_k_1 = zeros(n+1,1);

    z_k_K = zeros(n+1,1);
    sig_k_K = zeros(n+1,1);

    S = zeros(n,1);

    # batch state-space model outputs for the two attentional markers
    z = zeros(n,1);
    η = 0.3ones(n,1);

    ρ_d = ρ_d₀
    μ_d = μ_d₀

    P(x,i) = (1/x)*sqrt(ρ_d[i])*exp(-0.5ρ_d[i]*(log(x)-μ_d[i])^2)
    function outerE(x1,x2,z)
        p = sig(z)
        p*P(x1,1)*P(x2,1) /
            (p*P(x1,1)*P(x2,2) + (1-p)*(P(x1,2)*P(x2,2)))
    end
    function outerM(E,x1,x2,i)
        μ = ( sum(E.*log.(x1)+(1-E).*log.(x2)) + n*μ₀[i] )/2n
        ρ = (2n*α₀[i])/(
            sum( E.*((log.(x1).-μ_d[i]).^2) .+
                 (1.-E).*((log.(x2)-μ_d[i]).^2) ) +
            n*(2*β₀[i]+(μ_d[i]-μ₀[i]).^2)
        )
        μ,ρ
    end

    ########################################
    # outer EM
    for i in 1:outer_EM_batch
        # calculating epsilon_k's in the current iteration (E-Step)
        E = outerE.(x1,x2,z)

        # prior update
        μ_d[1], ρ_d[1] = outerM(E,x1,x2,1)
        μ_d[2], ρ_d[2] = outerM(E,x1,x2,2)

        ########################################
        # inner EM for updating z's and η's (M-Step)
        for j in 1:inner_EM_batch
            ##############################
            # inner E

            # filtering
            for t in 2:n+1
                z_k_k_1[t] = z_k_k[t-1];
                sig_k_k_1[t] = sig_k_k[t-1] + η[t-1];

                # Newton's algorithm
                for m=1:newton_iter
                    a = z_k_k[t] - z_k_k_1[t] - sig_k_k_1[t]*(E[t-1] -
                        exp(z_k_k[t])/(1+exp(z_k_k[t])))
                    b = 1 + sig_k_k_1[t]*exp(z_k_k[t])/((1+exp(z_k_k[t]))^2)
                    z_k_k[t] -= a / b;
                end
                sig_k_k[t] = 1 / (1/sig_k_k_1[t] +
                    exp(z_k_k[t])/((1+exp(z_k_k[t]))^2));
            end

            # smoothing
            z_k_K[n+1] = z_k_k[n+1];
            sig_k_K[n+1] = sig_k_K[n+1];

            for t = reverse(1:n)
                S[t] = sig_k_k[t]*1/sig_k_k_1[t+1];
                z_k_K[t] = z_k_k[t] + S[t]*(z_k_K[t+1] - z_k_k_1[t+1])
                sig_k_K[t] = sig_k_k[t] + S[t]^2*(sig_k_K[t+1] -
                    sig_k_k_1[t+1]);
            end

            z_k_k[1] = z_k_K[1]
            sig_k_k[1] = sig_k_K[1]

            ####################
            # inner M (update z and η)
            a = (z_k_K[2:end]-z_k_K[1:end-1]).^2 + sig_k_K[2:end] +
                sig_k_K[1:end-1]-2*sig_k_K[2:end].*S + 2*b₀
            η = a/(1+2*(a₀+1));

        end

        # update the z's
        z = z_k_K[2:end];
    end


end

end
