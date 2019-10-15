export StaticMethod, OnlineMethod, train_test, norm1, norm2
using LinearAlgebra

# TODO: we can have the speaker specific elements of the loop separated as part
# of a "TrainMethod" (we're going to have to come up with a name other than
# method, since this component is orthogonal to the static vs. online piece
# maybe TrainStimulus?

abstract type TrainMethod
end

struct StaticMethod{R,T} <: TrainMethod
    train::R
    test::T
end
StaticMethod(reg=EEGCoding.L2Matrix(0.2)) = StaticMethod(reg,cor)

label(x::StaticMethod) = join(("static",label(x.train)),"_")
test_label(x::StaticMethod) = join(("static",label(x.train),label(x.test)),"_")

label(x::NormL2) = "l2-$(x.lambda)"
label(x::NormL1) = "l1-$(x.lambda)"

norm1(x,y) = norm((xi - yi for (xi,yi) in zip(x,y)),1)/length(x)
norm2(x,y) = norm((xi - yi for (xi,yi) in zip(x,y)),2)/length(x)
label(x::Function) = string(x)

function init_result(::StaticMethod)
    DataFrame(sid = Int[],condition = String[], speaker = String[],
        trial = Int[], corr = Float64[],test_correct = Bool[])
end
train(m::StaticMethod;kwds...) = decoder(m.train;kwds...)
function test(m::StaticMethod;sid,condition,indices,sources,correct,
    kwds...)

    df, models = decode_test_cv(m.train,m.test;sources=sources,indices=indices,kwds...)

    df[!,:sid] .= sid
    df[!,:condition] .= condition
    df[!,:trial] = df.index
    df[!,:test_correct] = correct[df.index]

    models[!,:sid] .= sid
    models[!,:condition] .= condition

    df, models
end

struct OnlineMethod{S} <: TrainMethod
    params::S
end
OnlineMethod(;kwds...) = OnlineMethod(kwds.data)
label(::OnlineMethod) = "online"
Base.@kwdef struct OnlineResult
    sid::Int
    condition::String
    source::String
    test_correct::Bool
    trial::Int
    norms::Vector{Float64}
    probs::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
end

Tables.istable(::Type{<:AbstractVector{OnlineResult}}) = true
Tables.rowaccess(::Type{<:AbstractVector{OnlineResult}}) = true
Tables.rows(x::AbstractVector{OnlineResult}) = x
Tables.schema(x::AbstractVector{OnlineResult}) =
    Tables.Schema(fieldnames(OnlineResult),fieldtypes(OnlineResult))

function init_result(::OnlineMethod)
    Vector{OnlineResult}()
end
function train(::OnlineMethod;indices,kwds...)
    if length(indices) > 0
        error("Online algorithm uses no batch training, use an empty training set.")
    end
    nothing
end

function test(method::OnlineMethod;sid,condition,
    sources,indices,correct,model,kwds...)
    @assert isnothing(model)

    # @info "Call online decode"

    result = OnlineResult[]
    all_results = online_decode(;indices=indices,sources=sources,
        merge(kwds.data,method.params)...)
    for (trial_results,index,correct) in zip(all_results,indices,correct)
        for (source,(norms,probs,lower,upper)) in zip(sources,trial_results)
            result = push!(result,OnlineResult(
                trial = index,
                sid = sid,
                condition = condition,
                source = source,
                norms = norms,
                probs = probs,
                lower = lower,
                upper = upper,
                test_correct = correct
            ))
        end
    end
    result
end

function setup_indices(train_fn,test_fn,events)
    test_bounds = test_fn.(eachrow(events))
    train_bounds = train_fn.(eachrow(events))

    test_indices = findall(.!isempty.(test_bounds))
    train_indices = findall(.!isempty.(train_bounds))

    test_bounds, test_indices, train_bounds, train_indices
end

function train_test(method,stim_method,files,stim_info;
    maxlag=0.25,
    K=10,
    train = ["" => all_indices],
    test = train,
    resample = missing,
    encode_eeg = RawEncoding(),
    return_encodings = false,
    return_models = false,
    progress = true)

    result = init_result(method)
    train_sources, test_sources = sources(stim_method)
    n = 0
    for file in files
        events = events_for_eeg(file,stim_info)[1]
        for (traini,testi) in zip(train,test)
            test_bounds, test_indices,
                train_bounds, train_indices =
                    setup_indices(traini[2],testi[2],events)
            n += K*length(train_sources)
            n += K*length(test_sources)
        end
    end

    vcatdot(x,y) = vcat.(x,y)
    function maybe_parallel(fn,n,progress,files)
        if nprocs() > 1
            @info "Running on multiple child processes"
            parallel_progress(n,progress) do progress
                @distributed (vcatdot) for file in files
                    fn(file,progress)
                end
            end
        else
            @info "Running all file analyses in a single process."
            prog = (progress isa Bool && progress) ? Progress(n) : prog
            mapreduce(file -> fn(file,prog),vcatdot,files)
        end
    end

    maybe_parallel(n,progress,files) do file,progress
        eeg, stim_events, sid = load_subject(joinpath(data_dir(),file),stim_info,
            samplerate=resample,encoding=encode_eeg)
        lags = 0:round(Int,maxlag*samplerate(eeg))
        sid_str = @sprintf("%03d",sid)

        target_len = convert(Float64,stim_info["target_len"])

        mapreduce(vcatdot,zip(train,test)) do (traini,testi)
            test_bounds, test_indices, train_bounds, train_indices =
                setup_indices(traini[2],testi[2],stim_events)
            train_name = traini[1]
            test_name = testi[1]

            prefix = join([train_name, label(method), label(stim_method),
                string(encode_eeg), sid_str],"_")
            model = GermanTrack.train(method,
                K=K,
                sources = train_sources,
                prefix = prefix,
                eeg = eeg,
                lags=lags,
                indices = train_indices,
                bounds = train_bounds,
                progress = progress,
                stim_fn = load_source_fn(stim_method,stim_events,
                    coalesce(resample,samplerate(eeg)),stim_info)
            )

            test_prefix = join([test_name,test_label(method),
                label(stim_method),sid_str],"_")
            cond = string("train-",train_name,"__","test-",test_name)
            GermanTrack.test(method,
                K=K,
                sid = sid,
                condition = cond,
                sources = test_sources,
                train_source_indices = train_source_indices(stim_method),
                correct = stim_events.correct,
                prefix=test_prefix,
                train_prefix=prefix,
                eeg=eeg,
                model=model,
                lags=lags,
                indices = test_indices,
                bounds = test_bounds,
                progress = progress,
                stim_fn = load_source_fn(stim_method,stim_events,
                    coalesce(resample,samplerate(eeg)),stim_info,test=true)
            )
        end
    end
end
