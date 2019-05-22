# TODO: generify this method, so that it can be used to generate the results
# for either the static or the online analysis

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
    speaker::String
    test_correct::Bool
    trial::Int
    norms::Vector{Float64}
    probs::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
end

function init_result(::OnlineMethod)
    Vector{OnlineResult}()
end
function train(::OnlineMethod;indices,kwds...)
    if length(indices) > 0
        error("Online algorithm uses no batch training, use an empty training set.")
    end
    nothing
end

function test!(result,::OnlineMethod;bounds=all_indices,sid,condition,sources,
    indices,correct,model,kwds...)
    @assert isnothing(model)

    if any(x -> !(x == all_indices || x == no_indices),bounds)
        error("Online method does not currently support limited time ranges.")
    end

    all_results = online_decode(;indices=indices,sources=sources,
        merge(kwds,method.settings)...)
    for (trial_results,index,correct) in zip(all_results,indices,correct)
        for (source,(norms,probs,lower,upper)) in zip(sources,trial_results)
            result = push!(result,OnlineResult(
                trial = index,
                sid = sid,
                condition = condition,
                speaker = source,
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

# TODO: analyze all envelopes as one call to method, this will avoid some
# redundancy (e.g. calling `withlags` on eeg multiple times, and is more
# consistent with the interface for the online method)

function train_speakers(method,group_name,files,stim_info;
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

    speakers = ["male", "fem1", "fem2"]
    n = 0
    for file in files
        events = events_for_eeg(file,stim_info)[1]
        for cond in unique(events.condition)
            test_bounds, test_indices,
                train_bounds, train_indices = setup_indices(events,cond)
            n += length(train_indices)*length(speakers)
            n += length(test_indices)*(length(speakers)+1)
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

            prefix = join([train_name,label(method),cond,
                sid_str],"_")
            model = Main.train(method,
                sources = speakers,
                prefix = prefix,
                eeg = eeg,
                lags=lags,
                indices = train_indices,
                group_suffix = "_"*group_name,
                bounds = train_bounds,
                progress = progress,
                stim_fn = (i,j) -> load_sentence(stim_events,samplerate(eeg),
                    stim_info,i,j,envelope_method = envelope_method)
            )

            prefix = join([test_name,label(method),cond,
                sid_str],"_")
            Main.test!(result,method;
                sid = sid,
                condition = cond,
                sources = [speakers..., "male_other"],
                correct = stim_events.correct[test_indices],
                prefix=prefix,
                eeg=eeg,
                model=model,
                lags=lags,
                indices = test_indices,
                group_suffix = "_"*group_name,
                bounds = test_bounds,
                progress = progress,
                stim_fn = (i,j) -> j <= length(speakers) ?
                    load_sentence(stim_events,samplerate(eeg),stim_info,i,
                        j,envelope_method = envelope_method) :
                    load_other_sentence(stim_events,samplerate(eeg),stim_info,i,
                        1,envelope_method = envelope_method)
            )
        end
    end

    result
end
