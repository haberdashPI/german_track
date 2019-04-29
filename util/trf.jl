using ShammaModel

################################################################################
# handle selecting various bounds of a signal

# denotes selection of all valid time indices
struct AllIndices end
const all_indices = AllIndices()
Base.getindex(x::AllIndices,i::Int) = x
Base.isempty(x::AllIndices) = false
(x::AllIndices)(row) = all_indices

struct NoIndices end
const no_indices = NoIndices()
Base.isempty(x::NoIndices) = true

toindex(x,min,fs) = clamp.(round.(Int,x.*fs),1,min)

function select_bounds(x::AbstractArray,::AllIndices,min_len,fs,dim)
    if dim == 1
        x[1:min(min_len,end),:]
    elseif dim == 2
        x[:,1:min(min_len,end)]
    end
end

function select_bounds(x::MxArray,::AllIndices,min_len,fs,dim)
    if dim == 1
        mat"x = $x(1:min(end,$min_len),:);"
    elseif dim == 2
        mat"x = $x(:,1:min(end,$min_len));"
    end
    get_mvariable(:x)
end

function select_bounds(x::AbstractArray,(start,stop)::Tuple,min,fs,dim)
    start,stop = toindex.((start,stop),min,fs)
    if dim == 1
        x[start:stop,:]
    elseif dim ==2
        x[:,start:stop]
    else
        error("Unspported dimension $dim.")
    end
end

function select_bounds(x::MxArray,(start,stop)::Tuple,min,fs,dim)
    start,stop = toindex.((start,stop),min,fs)
    if dim == 1
        mat"x = $x($start:$stop,:);"
    elseif dim == 2
        mat"x = $x(:,$start:$stop);"
    else
        error("Unspported dimension $dim.")
    end

    get_mvariable(:x)
end


function select_bounds(x::AbstractArray,bounds::AbstractArray{<:Tuple},min,fs,dim)
    if dim == 1
        vcat(select_bounds.(Ref(x),bounds,min,fs,dim)...)
    elseif dim == 2
        hcat(select_bounds.(Ref(x),bounds,min,fs,dim)...)
    else
        error("Unspported dimension $dim.")
    end
end

function select_bounds(x::MxArray,bounds::AbstractArray{<:Tuple},min,fs,dim)
    mat"indices = [];"
    for (start,stop) in bounds
        start,stop = toindex.((start,stop),min,fs)
        mat"indices = [indices $start:$stop];"
    end
    if dim == 1
        mat"x = $x(indices,:);"
    elseif dim == 2
        mat"x = $x(:,indices);"
    else
        error("Unspported dimension $dim.")
    end

    get_mvariable(:x)
end

################################################################################
# testing and training

function trf_train(;prefix,group_suffix="",indices,name="Training",
    progress=Progress(length(indices),1,desc=name),kwds...)

    cachefn(@sprintf("%s_avg%s",prefix,group_suffix),
        trf_train_;prefix=prefix,indices=indices,name=name,progress=progress,
        oncache = () -> update!(progress,progress.counter+length(indices)),
        kwds...)
end

function trf_train_(;prefix,eeg,stim_info,lags,indices,stim_fn,name="Training",
        bounds=all_indices,progress=Progress(length(indices),1,desc=name))
    sum_model = Float64[]

    for i in indices
        stim = stim_fn(i)
        # for now, make signal monaural
        if size(stim,2) > 1
            stim = sum(stim,dims=2)
        end

        model = cachefn(@sprintf("%s_%02d",prefix,i),find_trf,stim,eeg,i,
            -1,lags,"Shrinkage",bounds[i])
        # model = find_trf(stim_envelope,response,-1,lags,"Shrinkage")
        if isempty(sum_model)
            sum_model = model
        else
            sum_model .+= model
        end
        next!(progress)
    end

    sum_model
end

# tried to make things faster by computing envelope in julia (to avoid copying
# the entire wav file): doesn't seem to matter much
function find_envelope(stim,tofs)
    N = round(Int,size(stim,1)/samplerate(stim)*tofs)
    result = zeros(N)
    window_size = 1.5/tofs
    toindex(t) = clamp(round(Int,t*samplerate(stim)),1,size(stim,1))

    for i in 1:N
        t = i/tofs
        from = toindex(t-window_size)
        to = toindex(t+window_size)
        result[i] = mean(x^2 for x in view(stim.data,from:to,:))
    end

    result
end

# function find_envelope(stim,tofs)
#     mat "result = CreateLoudnessFeature($(stim.data),$(samplerate(stim)),$tofs)"
#     get_mvariable(:result)
# end
function find_signals(stim,eeg,i,bounds=all_indices)
    # envelope and neural response
    fs = mat"$eeg.fsample"
    stim_envelope = find_envelope(stim,fs)

    mat"response = $eeg.trial{$i};"
    response = get_mvariable(:response)

    min_len = min(size(stim_envelope,1),trunc(Int,size(response,2)));

    stim_envelope = select_bounds(stim_envelope,bounds,min_len,fs,1)
    response = select_bounds(response,bounds,min_len,fs,2)

    stim_envelope,response
end

function find_trf(stim,eeg,i,dir,lags,method,bounds=all_indices;
        found_signals=nothing)
    if isnothing(found_signals)
        stim_envelope,response = find_signals(stim,eeg,i,bounds)
    else
        stim_envelope,response = found_signals
    end

    lags = collect(lags)
    mat"$result = FindTRF($stim_envelope,$response',-1,[],[],($lags)',$method);"
    result
end

function predict_trf(dir,response,model,lags,method)
    mat"[~,$result] = FindTRF([],[],-1,$response',$model,($lags)',$method);"
    result
end

function trf_corr_cv(;prefix,indices=indices,group_suffix="",name="Training",
    progress=Progress(length(indices),1,desc=name),kwds...)

    cachefn(@sprintf("%s_corr%s",prefix,group_suffix),
        trf_corr_cv_,;prefix=prefix,indices=indices,name=name,progress=progress,
        oncache = () -> update!(progress,progress.counter+length(indices)),
        kwds...)
end

function single(x)
    @assert(length(x) == 1)
    first(x)
end

function trf_corr_cv_(;prefix,eeg,stim_info,model,lags,indices,stim_fn,
    bounds=all_indices,name="Testing",
    progress=Progress(length(indices),1,desc=name))

    result = zeros(length(indices))

    for (j,i) in enumerate(indices)
        stim = stim_fn(i)
        if size(stim,2) > 1
            stim = sum(stim,dims=2)
        end
        stim_envelope,response = find_signals(stim,eeg,i,bounds[i])

        subj_model_file = joinpath(cache_dir,@sprintf("%s_%02d",prefix,i))
        # subj_model = load(subj_model_file,"contents")
        subj_model = cachefn(subj_model_file,find_trf,stim,eeg,i,-1,lags,
            "Shrinkage",bounds[i],found_signals = (stim_envelope,response))
        n = length(indices)
        r1, r2 = (n-1)/n, 1/n

        pred = predict_trf(-1,response,(r1.*model .- r2.*subj_model),lags,
            "Shrinkage")
        result[j] = single(cor(pred,stim_envelope))

        next!(progress)
    end
    result
end

function trf_train_speakers(group_name,files,stim_info;
    skip_bad_trials = false,
    maxlag=0.25,
    train = "" => all_indices,
    test = train)

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
            (events.correct) .& (.!isempty.(train_bounds)) .&
            (.!skip_bad_trials .| .!events.bad_trial))

        test_bounds, test_indices, train_bounds, train_indices
    end

    n = 0
    for file in files
        events = events_for_eeg(file,stim_info)[1]
        test_bounds, test_indices,
            train_bounds, train_indices = setup_indices(events,cond)
        n += length(test_indices)*3
        n += length(train_indices)*3
    end
    progress = Progress(n;desc="Analyzing...")

    for file in files
        eeg, stim_events, sid = load_subject(joinpath(data_dir,file),stim_info)
        lags = 0:round(Int,maxlag*mat"$eeg.fsample")

        target_len = convert(Float64,stim_info["target_len"])

        for cond in unique(stim_events.condition)
            test_bounds, test_indices,
             train_bounds, train_indices = setup_indices(stim_events,cond)

            sid_str = @sprintf("%03d",sid)

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
            end
        end
    end

    df
end
