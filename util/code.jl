using ProximalOperators
using ProximalAlgorithms

struct Objective <: ProximableFunction
    A::Matrix{Float64}
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
function update!(f::Objective,y,X,λ)
    f.A .*= λ
    f.A .+= X'X
    f.b .*= λ
    f.b .+= y'X
end
ProximalOperators.is_convex(::Objective) = true
ProximalOperators.is_smooth(::Objective) = true
ProximalOperators.is_separable(::Objective) = true
ProximalOperators.is_quadratic(::Objective) = true
ProximalOperators.is_strongly_convex(::Objective) = true
function (f::Objective)(x::AbstractArray)
    A,b = f.A,f.b
    y = x'A*x
    y -= 2 .* b*x
    sum(y)
end
function ProximalOperators.gradient!(y::AbstractArray,f::Objective,x::AbstractArray)
    A,b = f.A,f.b
    y .= 2 .* A*x .- 2 .* b'
    f(x)
end
ProximalOperators.fun_name(f::Objective) = "General Quadratic"
ProximalOperators.fun_dom(f::Objective) = "AbstractArray{Real}, AbstractArray{Real}"
ProximalOperators.fun_expr(f::Objective) = "x ↦ x'Ax - 2bx"
ProximalOperators.fun_params(f::Objective) = "A = $(f.A), b = $(f.b)"

function code(y,X,objective=Objective(y,X);λ=0.9,γ=1e-3,kwds...)
    update!(objective,y,X,λ)
    params = merge((maxit=1000, tol=1e-3, fast=true),kwds)
    _, solution = ProximalAlgorithms.FBS(objective.θ,fs=objective,fq=NormL1(γ);params...)
    objective.θ .= solution
    objective
end

