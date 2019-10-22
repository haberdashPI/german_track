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

apply_bounds(bounds,subject) = bounds.(eachrow(subject.events))

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

    # setup subjects
    @info "Loading subject data..."
    subjects = Dict(
        file => load_subject(joinpath(data_dir(),file),stim_info,
                             samplerate=resample,encoding=encode_eeg)
        for file in files
    )
    @info "Done!"

    # setup sources to train
    train_sources, test_sources = sources(stim_method)

    # count total number of K-fold tests across all conditions
    n = K*(length(train)*length(train_sources) +
        length(test)*length(test_sources))
    prog = (progress isa Bool && progress) ? Progress(n) : prog

    df, models = DataFrame(), DataFrame()

    fs = samplerate(first(values(subjects)).eeg)
    lags = round(Int,minlag*fs):round(Int,maxlag*fs)

    weights = Dict((file,i) => w
        for (file,subject) in subjects
        for (i,w) in enumerate(weightfn.(eachrow(subject.events))))

    function bound_indices(bounds,min)
        from,to = round.(Int,fs.*bounds)
        bound(from:to,min=1,max=min)
    end

    for (traini,testi) in zip(train,test)
        train_condition = traini[1]
        test_condition = testi[1]

        function train_prefix(source)
            join([values(train_condition);
                [string(maxlag), string(minlag), string(source),
                    label(method), label(stim_method), string(encode_eeg)]
            ],"_")
        end

        train_bounds = Dict((file,i) => bounds
            for file in files
            for (i,bounds) in enumerate(apply_bounds(traini[2],subjects[file])))
        train_indices =
            filter(@λ(!isempty(train_bounds[_])),keys(train_bounds)) |>
            collect |> sort!
        test_bounds = Dict((file,i) => bounds
            for file in files
            for (i,bounds) in enumerate(apply_bounds(testi[2],subjects[file])))
        test_indices =
            filter(@λ(!isempty(test_bounds[_])),keys(test_bounds)) |>
            collect |> sort!

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

                stims = mapreduce(vcat,enumerate(indices)) do (j,(file,i))
                    eeg,events = subjects[file]
                    stim, = load_stimulus(source,i,stim_method,events,
                        coalesce(resample,samplerate(eeg)),stim_info)
                    minlens[j] = min(size(stim,1),size(eeg[i],2))
                    times = bound_indices(train_bounds[(file,i)],minlens[j])

                    view(stim,times,:) .* weights[(file,i)]
                end

                responses = mapreduce(hcat,enumerate(indices)) do (j,(file,i))
                    eeg,events = subjects[file]

                    times = bound_indices(train_bounds[(file,i)],minlens[j])
                    view(eeg[i],:,times)
                end

                stims, responses'
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
            test_prefix = join([values(test_condition);
                [test_label(method),
                string(source),
                label(stim_method)]
            ],"_")
            cond = NamedTuple{(
                Symbol.("train_",keys(train_condition))...,
                Symbol.("test_",keys(test_condition))...)}(
                (values(train_condition)...,
                values(test_condition)...)
            )
            df_,models_ = decode_test_cv(method.train,method.test,
                K=K,
                prefix=test_prefix,
                train_prefix=prefix,
                lags=lags,
                indices = test_indices,
                train_indices = train_indices,
                progress = prog
            ) do (file,i)
                eeg, events = subjects[file]
                stim, stim_id = load_stimulus(source,i,stim_method,
                    events,coalesce(resample,samplerate(eeg)),
                    stim_info)

                minlen = min(size(stim,1),size(eeg[i],2))
                indices = bound_indices(test_bounds[(file,i)],minlen)
                stim = view(stim,indices,:) .* weights[(file,i)]
                response = view(eeg[i],:,indices)

                stim, response', stim_id
            end

            for key in keys(cond)
                df_[!,key] .= cond[key]
                models_[!,key] .= cond[key]
            end

            df_[!,:sid] .= sidfor.(getindex.(df_.index,1))
            df_[!,:trial] = getindex.(df_.index,2)
            names(df_)
            df_[!,:test_correct] = map(df_.index) do (file,i)
                subjects[file].events.correct[i]
            end
            select!(df_,Not(:index))
            df_[!,:source] .= string(source)

            models_[!,:source] .= string(source)

            df = vcat(df,df_)
            models = vcat(models,models_)
        end
    end

    df,models
end
