export StaticMethod, train_test, norm1, norm2
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

function setup_indices(train_fn,test_fn,events)
    test_bounds = test_fn.(eachrow(events))
    train_bounds = train_fn.(eachrow(events))

    test_indices = findall(.!isempty.(test_bounds))
    train_indices = findall(.!isempty.(train_bounds))

    test_bounds, test_indices, train_bounds, train_indices
end

function train_test(method,stim_method,files,stim_info;
    maxlag=0.25,
    minlag=0,
    K=10,
    train = ["" => all_indices],
    test = train,
    weightfn = events -> 1,
    resample = missing,
    encode_eeg = RawEncoding(),
    return_encodings = false,
    return_models = false,
    progress = true)

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

    df, models = DataFrame(), DataFrame()
    prog = (progress isa Bool && progress) ? Progress(n) : prog

    for file in files
        eeg, stim_events, sid = load_subject(joinpath(data_dir(),file),stim_info,
            samplerate=resample,encoding=encode_eeg)
        lags = round(Int,minlag*samplerate(eeg)):round(Int,maxlag*samplerate(eeg))
        sid_str = @sprintf("%03d",sid)

        weights = weightfn.(eachrow(stim_events))
        function bound_indices(bounds,min)
            from,to = round.(Int,samplerate(eeg).*bounds)
            bound(from:to,min=1,max=min)
        end

        for (traini,testi) in zip(train,test)
            test_bounds, test_indices, train_bounds, train_indices =
                setup_indices(traini[2],testi[2],stim_events)
            train_cond = traini[1]
            test_cond = testi[1]

            # TODO: more cleanup, I don't think I need the train
            # and test methods above

            function train_prefix(source)
                join([values(train_cond);
                    [string(maxlag), string(minlag), string(source),
                     label(method), label(stim_method), string(encode_eeg),
                     sid_str]
                ],"_")
            end

            for source in train_sources
                prefix = train_prefix(source)
                decoder(method.train,
                    K=K,
                    prefix = prefix,
                    lags=lags,
                    indices = train_indices,
                    progress = prog
                ) do indices
                    minlens = Vector{Int}(undef,length(indices))
                    stim_result = mapreduce(vcat,enumerate(indices)) do (j,i)
                        stim, = load_stimulus(source,i,stim_method,stim_events,
                            coalesce(resample,samplerate(eeg)),stim_info)
                        minlens[j] = min(size(stim,1),size(eeg[i],2))
                        times = bound_indices(train_bounds[i],minlens[j])

                        view(stim,times,:) .* weights[i]
                    end

                    response_result = mapreduce(hcat,enumerate(indices)) do (j,i)
                        times = bound_indices(train_bounds[i],minlens[j])
                        view(eeg[i],:,times)
                    end

                    stim_result, response_result'
                end
            end

            for source in test_sources
                # to fix the below problem: "other" sources
                # are for testing only, not training only
                # so my logic in the source refactoring is reversed
                # I should train a subset of sources and test over all_indices
                # sources, using the `fortraining` mehtod (now misnamed
                # `fortesting`)
                prefix = train_prefix(fortraining(source))
                test_prefix = join([values(test_cond);
                    [test_label(method),
                    string(source),
                    label(stim_method),sid_str]
                ],"_")
                cond = NamedTuple{(
                    Symbol.("train_",keys(train_cond))...,
                    Symbol.("test_",keys(test_cond))...)}(
                    (values(train_cond)...,
                    values(test_cond)...)
                )
                df_,models_ = decode_test_cv(method.train,method.test,
                    K=K,
                    prefix=test_prefix,
                    train_prefix=prefix,
                    lags=lags,
                    indices = test_indices,
                    train_indices = train_indices,
                    progress = prog
                ) do i
                    stim, stim_id = load_stimulus(source,i,stim_method,
                        stim_events,coalesce(resample,samplerate(eeg)),
                        stim_info)

                    minlen = min(size(stim,1),size(eeg[i],2))
                    indices = bound_indices(test_bounds[i],minlen)
                    stim = view(stim,indices,:) .* weights[i]
                    response = view(eeg[i],:,indices)

                    stim, response', stim_id
                end

                for key in keys(cond)
                    df_[!,key] .= cond[key]
                end

                df_[!,:trial] = df_.index
                df_[!,:test_correct] = stim_events.correct[df_.index]

                df_[!,:sid] .= sid
                df_[!,:source] .= string(source)
                models_[!,:sid] .= sid
                models_[!,:source] .= string(source)

                df = vcat(df,df_)
                models = vcat(models,models_)
            end
        end
    end

    df,models
end
