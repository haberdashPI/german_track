function trf_train_speakers(group_name,files,stim_info;
    skip_bad_trials = false,
    maxlag=0.25,
    train = "" => all_indices,
    test = train,
    envelope_method=:rms)

    train_name, train_fn = train
    test_name, test_fn = test

    df = DataFrame(sid = Int[],condition = String[], speaker = String[],
            corr = Float64[],test_correct = Bool[])

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
            n += length(test_indices)*(3+(cond == "male"))
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

                prefix = join([train_name,"trf",cond,speaker,sid_str],"_")
                model = trf_train(
                    prefix = prefix,
                    eeg = eeg,
                    stim_info = stim_info,lags=lags,
                    indices = train_indices,
                    group_suffix = "_"*group_name,
                    bounds = train_bounds,
                    progress = progress,
                    envelope_method = envelope_method,
                    stim_fn = i -> load_sentence(stim_events,stim_info,i,
                        speaker_index)
                )

                prefix = join([test_name,"trf",cond,speaker,sid_str],"_")
                C = trf_corr_cv(
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
                        speaker_index)
                )
                rows = DataFrame(
                    sid = sid,
                    condition = cond,
                    speaker=speaker,
                    corr = C,
                    test_correct = stim_events.correct[test_indices]
                )
                df = vcat(df,rows)

                if speaker == "male"
                    prefix = join([test_name,"trf",cond,"male_other",sid_str],"_")
                    C = trf_corr_cv(
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
                        stim_fn = i -> load_other_sentence(stim_events,stim_info,i,1)
                    )
                    rows = DataFrame(
                        sid = sid,
                        condition = cond,
                        speaker="male_other",
                        corr = C,
                        test_correct = stim_events.correct[test_indices]
                    )
                    df = vcat(df,rows)

                end
            end
        end

    end

    df
end
