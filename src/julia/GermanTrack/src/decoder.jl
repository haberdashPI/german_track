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

function lassoflux(x, y, λ, opt;
    batch = 64,
    validate = nothing,
    stop_threshold = 0.01,
    max_steps = 2,
    progress = Progress(steps))
    model = Dense(size(x, 1), size(y, 1)) |> gpu
    loss(x,y) = Flux.mse(model(x), y)

    l1opt = L1Opt(opt, λ, applyto = [model.W])

    local best_model
    best_loss = Float32(Inf32)
    stopped = false
    evalcb = if isnothing(validate)
        () -> nothing
        best_model = model
    else
        xᵥ, yᵥ = gpu.(validate)
        function ()
            cur_loss = loss(xᵥ, yᵥ)
            if cur_loss < best_loss
                best_loss = cur_loss
                best_model = deepcopy(model)
            end
            # @show best_loss
            # @show cur_loss
            if (cur_loss / best_loss - 1) > stop_threshold
                stopped = true
                @show cur_loss
                @show best_loss
                Flux.stop()
            end
        end
    end

    loader = Flux.Data.DataLoader((x |> gpu, y |> gpu), batchsize = batch, shuffle = true)
    local cur_step
    for outer cur_step in 1:max_steps
        Flux.Optimise.train!(loss, Flux.params(model), loader, l1opt)
        next!(progress)
        evalcb()
        stopped && break
    end
    for _ in (cur_step+1):max_steps
        next!(progress)
    end

    @show best_loss

    best_model |> cpu, cur_step
end
