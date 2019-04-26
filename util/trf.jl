using ShammaModel

# TODO: once this is basically working, convert to an all julia implementation:
# I don't really need the telluride toolbox to do what I'm doing right now and
# I can optimize it better if it's all in julia

function trf_train(prefix,args...;group_suffix="",kwds...)
    cachefn(@sprintf("%s_avg%s",prefix,group_suffix),
        trf_train_,prefix,args...;kwds...)
end

function trf_train_(prefix,eeg,stim_info,lags,indices,stim_fn;name="Training",
        bounds=nothing,progress=Progress(indices,1,descr=name))
    sum_model = Float64[]

    for i in indices
        stim = stim_fn(i)
        # for now, make signal monaural
        if size(stim,2) > 1
            stim = sum(stim,dims=2)
        end

        model = cachefn(@sprintf("%s_%02d",prefix,i),find_trf,stim,eeg,i,
            -1,lags,"Shrinkage",isnothing(bounds) ? nothing : bounds[i])
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

toindex(x,min,fs) = clamp.(round.(Int,x.*fs),1,min)

function select_bounds(x::AbstractArray,::Nothing,min_len,fs,dim)
    if dim == 1
        x[1:min(min_len,end),:]
    elseif dim == 2
        x[:,1:min(min_len,end)]
    end
end

function select_bounds(x::MxArray,::Nothing,min_len,fs,dim)
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

function find_signals(stim,eeg,i,bounds=nothing)
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

function find_trf(stim,eeg,i,dir,lags,method,bounds=nothing;
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

function trf_corr(eeg,stim_info,model,lags,indices,stim_fn;name="Testing",
    progress=Progress(indices,1,descr=name))

    result = zeros(length(indices))

    for (j,i) in enumerate(indices)
        stim = stim_fn(i)
        if size(stim,2) > 1
            stim = sum(stim,dims=2)
        end
        stim_envelope, response = find_signals(stim,eeg,i)

        pred = predict_trf(-1,response,model,lags,"Shrinkage")
        result[j] = cor(pred,stim_envelope)

        next!(progress)
    end
    result
end

function trf_corr_cv(prefix,args...;group_suffix="",kwds...)
    cachefn(@sprintf("%s_corr%s",prefix,group_suffix),
        trf_corr_cv_,prefix,args...;kwds...)
end

function single(x)
    @assert(length(x) == 1)
    first(x)
end

function trf_corr_cv_(prefix,eeg,stim_info,model,lags,indices,stim_fn;
        bounds=nothing,name="Testing")

    result = zeros(length(indices))

    @showprogress name for (j,i) in enumerate(indices)
        stim = stim_fn(i)
        if size(stim,2) > 1
            stim = sum(stim,dims=2)
        end
        bounds_i = isnothing(bounds) ? nothing : bounds[i]
        stim_envelope,response = find_signals(stim,eeg,i,bounds_i)

        subj_model_file = joinpath(cache_dir,@sprintf("%s_%02d",prefix,i))
        # subj_model = load(subj_model_file,"contents")
        subj_model = cachefn(subj_model_file,find_trf,stim,eeg,i,-1,lags,
            "Shrinkage",bounds_i,found_signals = (stim_envelope,response))
        n = length(indices)
        r1, r2 = (n-1)/n, 1/n

        pred = predict_trf(-1,response,(r1.*model .- r2.*subj_model),lags,
            "Shrinkage")
        result[j] = single(cor(pred,stim_envelope))
    end
    result
end

function trf_train_speakers(name,files,stim_info;
    stimulus_bounds=nothing,train_correct=false,maxlag=0.25)

    df = DataFrame(sid = Int[],condition = String[], speaker = String[],
            corr = Float64[],test_correct = Bool[])

    for file in files
        eeg, stim_events, sid = load_subject(joinpath(data_dir,file))
        lags = 0:round(Int,maxlag*mat"$eeg.fsample")
        seed = hash(eeg_file)

        target_times = convert(Array{Float64},
            stim_info["test_block_cfg"]["target_times"][stim_events.sound_index])
        stim_events[:target_present] = target_times .> 0
        stim_events[:correct] = stim_events.target_present .==
            (stim_events.response .== 2)

        stim_events[:target_time] = target_times
        target_len = convert(Float64,stim_info["target_len"])

        for cond in unique(stim_events.condition)
            test = findall(stim_events.condition .== cond)
            train = findall((stim_events.condition .== cond) .&
                (stim_events.correct))

            bounds = isnothing(stimulus_bounds) ? nothing :
                stimulus_bounds[stim_events.sound_index]

            # TODO: where I left off: insert progress bar for everything...
            male_model = trf_train(@sprintf("trf_%s_male_%s_sid_%03d",cond,name,sid),
                eeg,stim_info,lags,indices,
                name = @sprintf("Training SID %02d (Male): ",sid),
                bounds = bounds,
                i -> load_sentence(stim_events,stim_info,i,male_index))

            fem1_model = trf_train(@sprintf("trf_%s_fem1_switch_sid_%03d",cond,sid),
                eeg,stim_info,lags,indices,
                name = @sprintf("Training SID %02d (Female 1): ",sid),
                bounds = bounds,
                i -> load_sentence(stim_events,stim_info,i,fem1_index))

            fem2_model = trf_train(@sprintf("trf_%s_fem2_switch_sid_%03d",cond,sid),
                eeg,stim_info,lags,indices,
                name = @sprintf("Training SID %02d (Female 2): ",sid),
                bounds = bounds,
                i -> load_sentence(stim_events,stim_info,i,fem2_index))

            # should these also be bounded by the target?

            C = trf_corr_cv(@sprintf("trf_%s_male_switch_sid_%s03d",cond,sid),eeg,
                    stim_info,male_model,lags,indices,
                    name = @sprintf("Testing SID %02d (Male): ",sid),
                    bounds = bounds,
                    i -> load_sentence(stim_events,stim_info,i,male_index))
            df = vcat(df,DataFrame(sid = sid, condition = cond,
                    speaker="male", corr = C,
                    test_correct = stim_events.correct[indices]))

            C = trf_corr_cv(@sprintf("trf_%s_fem1_switch_sid%03d",cond,sid),eeg,
                    stim_info,fem1_model,lags,indices,
                    name = @sprintf("Testing SID %02d (Female 1): ",sid),
                    bounds = bounds,
                    i -> load_sentence(stim_events,stim_info,i,fem1_index))
            df = vcat(df,DataFrame(sid = sid, condition = cond,
                    speaker="fem1", corr = C,
                    test_correct = stim_events.correct[indices]))

            C = trf_corr_cv(@sprintf("trf_%s_fem2_switch_sid%03d",cond,sid),eeg,
                    stim_info,fem2_model,lags,indices,
                    name = @sprintf("Testing SID %02d (Female 2): ",sid),
                    bounds = bounds,
                    i -> load_sentence(stim_events,stim_info,i,fem2_index))
                df = vcat(df,DataFrame(sid = sid, condition = cond,
                    speaker="fem2", corr = C,
                    test_correct = stim_events.correct[indices]))

            C = trf_corr_cv(@sprintf("trf_%s_male_switch_sid%03d",cond,sid),eeg,
                    stim_info,male_model,lags,indices,
                    name = @sprintf("Testing SID %02d (Other Male): ",sid),
                    group_suffix = "_other",
                    bounds = bounds,
                    i -> load_other_sentence(stim_events,stim_info,i,male_index))
            df = vcat(df,DataFrame(sid = sid, condition = cond,
                    speaker="other_male", corr = C,
                    test_correct = stim_events.correct[indices]))
