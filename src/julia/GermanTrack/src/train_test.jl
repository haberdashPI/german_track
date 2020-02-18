export StaticMethod, train_test, norm1, norm2, apply_bounds, bound_indices,
    load_subject
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
StaticMethod(reg=NormL2(0.2)) = StaticMethod(reg,cor)

label(x::StaticMethod) = join(("static",label(x.train)),"_")
test_label(x::StaticMethod) = join(("static",label(x.train),label(x.test)),"_")

label(x::NormL2) = "l2-$(x.lambda)"
label(x::NormL1) = "l1-$(x.lambda)"

norm1(x,y) = norm((xi - yi for (xi,yi) in zip(x,y)),1)/length(x)
norm2(x,y) = norm((xi - yi for (xi,yi) in zip(x,y)),2)/length(x)
label(x::Function) = string(x)

apply_bounds(bounds,subject) = bounds.(eachrow(subject.events))
function bound_indices(bounds,fs,min)
    from,to = round.(Int,fs.*bounds)
    bound(from:to,min=1,max=min)
end

function EEGCoding.cleanstring((file,i)::Tuple{String,Int})
    @sprintf("%02d_%03d",sidfor(file),i)
end

function train_test(method,stim_method,files,stim_info;
    maxlag=0.25,
    minlag=0,
    train = ["" => all_indices],
    test = train,
    weightfn = events -> 1,
    resample = missing,
    encode_eeg = RawEncoding(),
    return_encodings = false,
    progress = true,
    subjects = nothing)

    # setup subjects
    if isnothing(subjects)
        @info "Loading subject data..."
        subjects = Dict(
            file => load_subject(joinpath(data_dir(),file),stim_info,
                                framerate=resample,encoding=encode_eeg)
            for file in files
        )
        @info "Done!"
    end

    # setup sources to train
    train_sources, test_sources = sources(stim_method)

    # count total number of K-fold tests across all conditions
    n = sum(zip(train,test)) do (traini,testi)
        sum(values(subjects)) do subject
            sum(!isempty,apply_bounds(traini[2],subject)) +
            sum(!isempty,apply_bounds(testi[2],subject))
        end
    end
    prog = (progress isa Bool && progress) ? Progress(n) : prog

    df = DataFrame()

    fs = framerate(first(values(subjects)).eeg)
    lags = round(Int,minlag*fs):round(Int,maxlag*fs)

    weights = Dict((file,i) => w
        for (file,subject) in subjects
        for (i,w) in enumerate(weightfn.(eachrow(subject.events))))


    for (traini,testi) in zip(train,test)
        train_condition = traini[1]
        test_condition = testi[1]

        function train_prefix(source)
            join([
                values(train_condition)...,
                string(maxlag), string(minlag), string(source),
                label(method), label(stim_method), string(encode_eeg)
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
                prefix = prefix,
                lags=lags,
                indices = train_indices,
                progress = prog
            ) do (file,i)

                eeg,events = subjects[file]
                full_stim, = load_stimulus(source,i,stim_method,events,
                    coalesce(resample,framerate(eeg)),stim_info)
                minlen = min(size(full_stim,1),size(eeg[i],2))
                times = bound_indices(train_bounds[(file,i)],framerate(eeg),
                    minlen)
                stim = view(full_stim,times,:) .* weights[(file,i)]
                response = view(eeg[i],:,times)

                stim, response'
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
            test_prefix = join([
                values(test_condition)...,
                test_label(method),
                string(source),
                label(stim_method)
            ],"_")
            cond = NamedTuple{(
                Symbol.("train_",keys(train_condition))...,
                Symbol.("test_",keys(test_condition))...)}(
                (values(train_condition)...,
                values(test_condition)...)
            )
            df_ = testdecode(method.train,method.test,
                prefix=test_prefix,
                train_prefix=prefix,
                lags=lags,
                indices = test_indices,
                train_indices = train_indices,
                progress = prog
            ) do (file,i)
                eeg, events = subjects[file]
                stim, stim_id = load_stimulus(source,i,stim_method,
                    events,coalesce(resample,framerate(eeg)),
                    stim_info)

                minlen = min(size(stim,1),size(eeg[i],2))
                indices = bound_indices(test_bounds[(file,i)],framerate(eeg),
                    minlen)
                stim = view(stim,indices,:) .* weights[(file,i)]
                response = view(eeg[i],:,indices)

                stim, response', stim_id
            end

            for key in keys(cond)
                df_[!,key] .= cond[key]
            end

            df_[!,:sid] .= sidfor.(getindex.(df_.index,1))
            df_[!,:trial] = getindex.(df_.index,2)
            names(df_)
            df_[!,:test_correct] = map(df_.index) do (file,i)
                subjects[file].events.correct[i]
            end
            select!(df_,Not(:index))
            df_[!,:source] .= string(source)
            df = vcat(df,df_)
        end
    end

    df
end
