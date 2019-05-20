# TODO: generify this method, so that it can be used to generate the results
# for either the static or the online analysis

abstract type TrainMethod; end

struct StaticMethod <: TrainMethod; end
label(::StaticMethod) = "trf" # legacy name, should be changed in the future
function init_result(::StaticMethod)
    DataFrame(sid = Int[],condition = String[], speaker = String[],
        corr = Float64[],test_correct = Bool[])
end
train(::StaticMethod;kwds...) = trf_train(;kwds...)
function test(result,::StaticMethod;sid,condition,speaker,correct,kwds...)
    C = trf_corr_cv(;kwds...)

    vcat(result,DataFrame(
        sid = sid,
        condition = cond,
        speaker = speaker,
        corr = C,
        test_correct = correct
    ))
end

struct OnlineMethod <: TrainMethod; end
label(::OnlineMethod) = "online"
struct OnlineResult
    sid::Int
    condition::String
    speaker::String
    test_correct::Bool
    norms::Vector{Float64}
end

function init_result(::OnlineMethod)
    Array{OnlineResult}()
end
train(::OnlineMethod;kwds...) = nothing
function test(result,::OnlineMethod;bounds=all_indices,sid,condition,speaker,
    correct,kwds...)
    if bounds !== all_indices
        error("Online method does not currently support limited time range.")
    end

    coefs = online_decode(;kwds...)

    vcat(result,OnlineResult(
        sid = sid,
        condition = cond,
        speaker = speaker,
        norms =
        test_correct = correct
    ))
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

    n = 0
    for file in files
        events = events_for_eeg(file,stim_info)[1]
        for cond in unique(events.condition)
            test_bounds, test_indices,
                train_bounds, train_indices = setup_indices(events,cond)
            n += length(train_indices)*3
            n += length(test_indices)*4
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

            for (speaker_index,speaker) in enumerate(["male", "fem1", "fem2"])

                prefix = join([train_name,label(method),cond,speaker,
                    sid_str],"_")
                model = Main.train(method,
                    prefix = prefix,
                    eeg = eeg,
                    stim_info = stim_info,lags=lags,
                    indices = train_indices,
                    group_suffix = "_"*group_name,
                    bounds = train_bounds,
                    progress = progress,
                    stim_fn = i -> load_sentence(stim_events,stim_info,i,
                        speaker_index,envelope_method = envelope_method)
                )

                prefix = join([test_name,label(method),cond,speaker,
                    sid_str],"_")
                result = Main.test(result,method;
                    sid = sid,
                    condition = cond,
                    speaker = speaker,
                    correct = stim_events.correct[test_indices],
                    prefix=prefix,
                    eeg=eeg,
                    stim_info=stim_info,
                    model=model,
                    lags=lags,
                    indices = test_indices,
                    group_suffix = "_"*group_name,
                    bounds = test_bounds,
                    progress = progress,
                    envelope_method = envelope_method,
                    stim_fn = i -> load_sentence(stim_events,stim_info,i,
                        speaker_index,envelope_method = envelope_method)
                )

                if speaker == "male"
                    prefix = join([test_name,label(method),cond,"male_other",
                        sid_str],"_")
                    result = Main.test(result,method;
                        sid = sid,
                        condition = cond,
                        speaker="male_other",
                        correct = stim_events.correct[test_indices],
                        prefix=prefix,
                        eeg=eeg,
                        stim_info=stim_info,
                        model=model,
                        lags=lags,
                        indices = test_indices,
                        group_suffix = "_"*group_name,
                        bounds = test_bounds,
                        progress = progress,
                        envelope_method = envelope_method,
                        stim_fn = i -> load_other_sentence(stim_events,stim_info,i,
                            speaker_index,envelope_method = envelope_method)
                    )
                end
            end
        end

    end

    df
end
