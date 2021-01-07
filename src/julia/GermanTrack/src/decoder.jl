export lassoflux

struct L1Opt{O,F}
    opt::O
    lambda::F
    state::IdDict{Any,Any}
    applyto::IdDict{Any,Bool}
end
function L1Opt(opt, lambda; applyto = [])
    at = IdDict{Any,Bool}()
    for a in applyto
        at[a] = true
    end
    L1Opt(opt, lambda, IdDict(), at)
end

# use the proximal operator for l1 to shrink weights towards 0
# only apply to parameters of the model for which this is requested
# (e.g. we don't apply it to the bias terms)
function Flux.Optimise.update!(o::L1Opt, x::AbstractArray, Δ::AbstractArray)
    if get(o.applyto, x, false)
        Δ₀ = get!(() -> similar(Δ), o.state, x)::typeof(Δ)
        Δ₀ .= Δ
        Flux.Optimise.update!(o.opt, x, Δ)
        η = abs.(Δ ./ Δ₀)
        x .= sign.(x) .* max.(0.0f0, abs.(x) .- η.*eltype(x)(o.lambda))
    else
        Flux.Optimise.update!(o.opt, x, Δ)
    end
end

function lassoflux(x, y, λ, opt, steps; batch = 64, progress = Progress(steps))
    model = Dense(size(x, 1), size(y, 1)) |> gpu
    loss(x,y) = Flux.mse(model(x), y)

    l1opt = L1Opt(opt, λ, applyto = [model.W])

    loader = Flux.Data.DataLoader((x |> gpu, y |> gpu), batchsize = batch, shuffle = true)
    for _ in 1:steps
        Flux.Optimise.train!(loss, Flux.params(model), loader, l1opt)
        next!(progress)
    end

    model |> cpu
end
