
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

struct Decoder{M}
    model::M
    steps::Int
end
nsteps(x::Decoder) = x.steps
(d::Decoder)(x) = d.model(x)
decode_weights(x::Decoder) = mean(x.model.layers[1].W, dims = 1)
Flux.gpu(x::Decoder) = Decoder(gpu(x.model), x.steps)

function decoder(x, y, λ, opt;
    batch = 64,
    validate = nothing,
    patience = 0,
    max_steps = 2,
    min_steps = 1,
    max_x_memory = 8(2^10)^3,
    inner = 1024,
    progress = Progress(steps))


    model = Chain(
        Dense(size(x, 1), inner),
        BatchNorm(inner, swish),
        Dense(inner, size(y, 1)),
    ) |> gpu

    λf = Float32(λ)
    loss(x,y) = Flux.mse(model(x), y) #.- λf.*sum(abs, decode_weights(model))

    l1opt = L1Opt(opt, λ, applyto = [model.layers[1].W])
    # l1opt = opt

    local best_model = deepcopy(model)
    cur_step = 0

    best_loss = Float32(Inf32)
    best_steps = 0
    stopped = false
    evalcb = if isnothing(validate)
        () -> nothing
        best_model = model
    else
        xᵥ, yᵥ = gpu.(validate)
        wait = 1
        function ()
            cur_loss = loss(xᵥ, yᵥ)
            if cur_loss < best_loss
                wait = 0
                best_loss = cur_loss
                best_steps = cur_step
                best_model = deepcopy(model)
            end
            # @show best_loss
            # @show cur_loss
            if cur_loss > best_loss
                wait += 1
                if wait > patience && min_steps < cur_step
                    stopped = true
                    # Flux.stop()
                end
            end
        end
    end

    loader = if true #sizeof(x) > max_x_memory
        @warn "Large data set: $(round(sizeof(x) / (2^10)^3, digits=2)) GB. Loading into memory in batches." maxlog=1 _id=objectid(x)
        Iterators.map(Flux.Data.DataLoader((x, y), batchsize = batch, shuffle = true)) do (x_batch, y_batch)
            (x_batch |> gpu, y_batch |> gpu)
        end
    else
        Flux.Data.DataLoader((x |> gpu, y |> gpu), batchsize = batch, shuffle = true)
    end
    for outer cur_step in 1:max_steps
        Flux.Optimise.train!(loss, Flux.params(model), loader, l1opt)
        next!(progress)
        evalcb()
        stopped && break
    end

    for _ in (cur_step+1):max_steps
        next!(progress)
    end

    Decoder(testmode!(best_model |> cpu), best_steps)
end
