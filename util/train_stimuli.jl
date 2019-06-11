# TODO: we can have the speaker specific elements of the loop separated as part
# of a "TrainMethod" (we're going to have to come up with a name other than
# method, since this component is orthogonal to the static vs. online piece
# maybe TrainStimulus?

abstract type StimMethod
end

Base.@kwdef struct SpeakerStimMethod <: StimMethod
    envelope_method::Symbol
end
label(x::SpeakerStimMethod) = "speakers_"*string(x.envelope_method)
sources(::SpeakerStimMethod) =
    ["male", "fem1", "fem2"], ["male", "fem1", "fem2", "male_other"]
function load_source_fn(method::SpeakerStimMethod,stim_events,fs,stim_info)
    function(i,j)
        if j <= 3
            load_speaker(stim_events,fs,i,j,
                envelope_method=method.envelope_method)
        else
            load_other_speaker(stim_events,fs,stim_info,i,1,
                envelope_method=method.envelope_method)
        end
    end
end

Base.@kwdef struct ChannelStimMethod <: StimMethod
    envelope_method::Symbol
end
label(x::ChannelStimMethod) = "channels_"*string(x.envelope_method)
sources(::ChannelStimMethod) = ["left", "right"], ["left", "right", "left_other"]
function load_source_fn(method::ChannelStimMethod,stim_events,fs,stim_info)
    function(i,j)
        if j <= 2
            load_channel(stim_events,fs,i,j,
                envelope_method=method.envelope_method)
        else
            load_other_channel(stim_events,fs,stim_info,i,1,
                envelope_method=method.envelope_method)
        end
    end
end


abstract type TrainMethod
end

struct StaticMethod <: TrainMethod
end

label(::StaticMethod) = "static"
function init_result(::StaticMethod)
    DataFrame(sid = Int[],condition = String[], speaker = String[],
        trial = Int[], corr = Float64[],test_correct = Bool[])
end
train(::StaticMethod;kwds...) = trf_train(;kwds...)
function test!(result,::StaticMethod;sid,condition,indices,sources,correct,
    kwds...)

    C = trf_corr_cv(;sources=sources,indices=indices,kwds...)

    append!(result,DataFrame(
        sid = sid,
        condition = cond,
        trial = indices,
        speaker = repeat(sources,outer=length(indices)),
        corr = C,
        test_correct = correct
    ))
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

function test!(result,method::OnlineMethod;sid,condition,
    sources,indices,correct,model,kwds...)
    @assert isnothing(model)

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


function train_stimuli(method,stim_method,files,stim_info;
    skip_bad_trials = false,
    maxlag=0.25,
    train = "" => all_indices,
    test = train,
    envelope_method=:rms)

    train_name, train_fn = train
    test_name, test_fn = test

    result = init_result(method)

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

    progress = Progress(n;desc="Analyzing...")

    for file in files
        # TODO: this relies on experiment specific details how to generify
        # this (or should we just move this whole function back)?
        eeg, stim_events, sid = load_subject(joinpath(data_dir,file),stim_info)
        lags = 0:round(Int,maxlag*samplerate(eeg))
        sid_str = @sprintf("%03d",sid)

        target_len = convert(Float64,stim_info["target_len"])

        for cond in unique(stim_events.condition)
            test_bounds, test_indices,
             train_bounds, train_indices = setup_indices(stim_events,cond)

            prefix = join([train_name,!skip_bad_trials ? "bad" : "",
                label(method),label(stim_method),cond, sid_str],"_")
            model = Main.train(method,
                sources = train_sources,
                prefix = prefix,
                eeg = eeg,
                lags=lags,
                indices = train_indices,
                bounds = train_bounds,
                progress = progress,
                stim_fn = load_source_fn(stim_method,stim_events,
                    samplerate(eeg),stim_info)
            )

            prefix = join([test_name,!skip_bad_trials ? "bad" : "",
                label(method),label(stim_method),cond,sid_str],"_")
            Main.test!(result,method;
                sid = sid,
                condition = cond,
                sources = test_sources,
                correct = stim_events.correct[test_indices],
                prefix=prefix,
                eeg=eeg,
                model=model,
                lags=lags,
                indices = test_indices,
                bounds = test_bounds,
                progress = progress,
                stim_fn = load_source_fn(stim_method,stim_events,
                    samplerate(eeg),stim_info)
            )
        end
    end

    result
end
