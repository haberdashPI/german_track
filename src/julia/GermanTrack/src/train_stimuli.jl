export StaticMethod, SpeakerStimMethod, ChannelStimMethod, OnlineMethod,
    train_stimuli, norm1, norm2
using LinearAlgebra

# TODO: we can have the speaker specific elements of the loop separated as part
# of a "TrainMethod" (we're going to have to come up with a name other than
# method, since this component is orthogonal to the static vs. online piece
# maybe TrainStimulus?

abstract type StimMethod
end

Base.@kwdef struct SpeakerStimMethod <: StimMethod
    encoding::EEGCoding.Encoding
end
label(x::SpeakerStimMethod) = "speakers_"*string(x.encoding)
sources(::SpeakerStimMethod) =
    ["male", "fem1", "fem2", "all-male"],
        ["male", "fem1", "fem2", "all-male", "male_other"]
train_source_indices(::SpeakerStimMethod) = (1,2,3,4,1)
function load_source_fn(method::SpeakerStimMethod,stim_events,fs,stim_info;
    test=false)

    function(i,j)
        if 0 < j <= 3
            load_speaker(stim_events,fs,i,j,
                encoding=method.encoding)
        elseif j == 4
            load_speaker_mix_minus(stim_events,fs,i,1,
                encoding=method.encoding)
        elseif j == 5
            if !test
                load_speaker(stim_events,fs,i,1,
                    encoding=method.encoding)
            else
                load_other_speaker(stim_events,fs,stim_info,i,1,
                    encoding=method.encoding)
            end
        else
            error("Did not expect j == $j.")
        end
    end
end

Base.@kwdef struct ChannelStimMethod <: StimMethod
    encoding::Symbol
end
label(x::ChannelStimMethod) = "channels_"*string(x.encoding)
sources(::ChannelStimMethod) =
    ["left", "right"], ["left", "right", "left_other"]
train_source_indices(::ChannelStimMethod) = (1,2,1)
function load_source_fn(method::ChannelStimMethod,stim_events,fs,stim_info)
    function(i,j)
        if j <= 2
            load_channel(stim_events,fs,i,j,
                encoding=method.encoding)
        else
            load_other_channel(stim_events,fs,stim_info,i,1,
                encoding=method.encoding)
        end
    end
end


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
label(x::typeof(norm1)) = "norm1"
label(x::typeof(norm2)) = "norm2"
label(x::typeof(cor)) = "cor"

function init_result(::StaticMethod)
    DataFrame(sid = Int[],condition = String[], speaker = String[],
        trial = Int[], corr = Float64[],test_correct = Bool[])
end
train(m::StaticMethod;kwds...) = decoder(m.train;kwds...)
function test(m::StaticMethod;sid,condition,indices,sources,correct,
    kwds...)

    speaker = decode_test_cv(m.train,m.test;sources=sources,indices=indices,kwds...)
    speaker[!,:sid] .= sid
    speaker[!,:condition] .= condition
    speaker[!,:trial] = indices[speaker.index]
    speaker[!,:test_correct] = correct[speaker.index]

    speaker
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

#= function train_stimulus_steps(method,stim_method,files,stim_info;
    skip_bad_trials = false)

    train_sources, test_sources = sources(stim_method)
    n = 0
    for file in files
        events = events_for_eeg(file,stim_info)[1]
        for cond in unique(events.condition)
            test_bounds, test_indices,
                train_bounds, train_indices = setup_indices(events,cond)
            n += length(train_indices)*length(train_sources)
            n += length(test_indices)*length(test_sources)
        end
    end

    function setup_indices(events,cond)
        test_bounds = test_fn.(eachrow(events))
        train_bounds = train_fn.(eachrow(events))

        test_indices = findall((events.condition .== cond) .&
            (.!isempty.(test_bounds)) .&
            (.!skip_bad_trials .| .!events.bad_trial))
        train_indices = findall((events.condition .== cond) .&
            (.!isempty.(train_bounds)) .&
            (.!skip_bad_trials .| .!events.bad_trial))

        test_bounds, test_indices, train_bounds, train_indices
    end
end =#

function setup_indices(train_fn,test_fn,events)
    test_bounds = test_fn.(eachrow(events))
    train_bounds = train_fn.(eachrow(events))

    test_indices = findall(.!isempty.(test_bounds))
    train_indices = findall(.!isempty.(train_bounds))

    test_bounds, test_indices, train_bounds, train_indices
end

function train_stimuli(method,stim_method,files,stim_info;
    skip_bad_trials = false,
    maxlag=0.25,
    train = ["" => all_indices],
    test = train,
    resample = missing,
    encode_eeg = RawEncoding(),
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
            n += length(train_indices)*length(train_sources)
            n += length(test_indices)*length(test_sources)
        end
    end

    # @info "HELLO!"

    function maybe_parallel(fn,n,progress,files)
        if nprocs() > 1
            @info "Running on multiple child processes"
            parallel_progress(n,progress) do progress
                @distributed (vcat) for file in files
                    fn(file,progress)
                end
            end
        else
            @info "Running all file analyses in a single process."
            prog = (progress isa Bool && progress) ? Progress(n) : prog
            mapreduce(file -> fn(file,prog),vcat,files)
        end
    end

    maybe_parallel(n,progress,files) do file,progress
        global data_dir
        eeg, stim_events, sid = load_subject(joinpath(data_dir(),file),stim_info,
            samplerate=resample,encoding=encode_eeg)
        lags = 0:round(Int,maxlag*samplerate(eeg))
        sid_str = @sprintf("%03d",sid)

        target_len = convert(Float64,stim_info["target_len"])

        mapreduce(vcat,zip(train,test)) do (traini,testi)
            test_bounds, test_indices, train_bounds, train_indices =
                setup_indices(traini[2],testi[2],stim_events)
            train_name = traini[1]
            test_name = testi[1]

            prefix = join([train_name, label(method), label(stim_method),
                string(encode_eeg), sid_str],"_")
            model = GermanTrack.train(method,
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
            @show model[1][1:5]

            test_prefix = join([test_name,test_label(method),
                label(stim_method),sid_str],"_")
            @show test_prefix
            GermanTrack.test(method;
                sid = sid,
                condition = string("train",train_name,"_","test",test_name),
                sources = test_sources,
                train_source_indices = train_source_indices(stim_method),
                correct = stim_events.correct[test_indices],
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
