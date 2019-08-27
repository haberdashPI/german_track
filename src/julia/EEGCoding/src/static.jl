export withlags, decode_test_cv, decoder, RegNorm

using MetaArrays
using Printf
using DataFrames
using StatsBase
using Statistics
using CorticalSpectralTemporalResponses
using DSP
using ProximalAlgorithms
using ProximalOperators

# custom decoding method (others use StructuedOptimizations data types)
struct L2Matrix
    k::Float64
end

################################################################################
# testing and training

function decoder(method;prefix,group_suffix="",indices,name="Training",
    sources,progress=Progress(length(indices)*length(sources),1,desc=name),
    kwds...)

    cachefn(@sprintf("%s_avg%s",prefix,group_suffix),
        decoder,method;prefix=prefix,indices=indices,name=name,progress=progress,
        sources=sources,
        __oncache__ = () ->
            progress_update!(progress,length(indices)*length(sources)),
        kwds...)
end

function decoder(method;prefix,eeg,lags,indices,stim_fn,name="Training",
        sources,bounds=all_indices,progress,kwds...)

    sum_models = [Array{Float64}(undef,0,0,0) for i in 1:length(sources)]

    for i in indices
        # TODO: implement trial_decoder to handle multiple stimuli at once? (reduces
        # slow repeated call to withlags) PROFILE first (I think most of the
        # time is spent computing the regression)
        for (source_index,source) in enumerate(sources)
            stim = stim_fn(i,source_index)

            model = cachefn(@sprintf("%s_%s_%02d",source,prefix,i),
                trial_decoder,method,stim,eeg,i,lags;bounds=bounds[i],kwds...)

            if isempty(sum_models[source_index])
                sum_models[source_index] = model
            else
                sum_models[source_index] .+= model
            end
            progress_update!(progress)
        end
    end

    sum_models
end

find_signals(found_signals,stim,eeg,i;kwds...) = found_signals
function find_signals(::Nothing,stim,eeg,i;bounds=all_indices)
    response = eegtrial(eeg,i)
    min_len = min(size(stim,1),trunc(Int,size(response,2)));

    stim = select_bounds(stim,bounds,min_len,samplerate(eeg),1)
    response = select_bounds(response,bounds,min_len,samplerate(eeg),2)

    stim,response
end

function withlags(x,lags)
    if lags == 0:0
        x
    end

    nl = length(lags)
    n,m = size(x)
    y = similar(x,size(x,1),m*nl)
    z = zero(eltype(y))
    for I in CartesianIndices(x)
        for (l,lag) in enumerate(lags)
            r,c = I[1],I[2]
            r_ = r - lag
            y[r,(l-1)*m+c] = 0 < r_ <= n ? x[r_,c] : z
        end
    end
    y
end

# TODO: we could probably make things even faster if we created the memory XX
# and XY once.

scale(x) = mapslices(zscore,x,dims=1)
# adds v to the diagonal of matrix (or tensor) x
adddiag!(x,v) = x[CartesianIndex.(axes(x)...)] .+= v
function trial_decoder(l2::L2Matrix,stim,eeg::EEGData,i,lags;
    found_signals=nothing, kwds...)

    stim,response = find_signals(found_signals,stim,eeg,i;kwds...)

    X = withlags(scale(response'),.-reverse(lags))
    Y = scale(stim)

    k = l2.k
    XX = X'X; XY = Y'X
    λ̄ = tr(XX)/size(X,2)
    XX .*= (1-k); adddiag!(XX,k*λ̄)
    result = XX\XY' # TODO: in Julia 1.2, this can probably be replaced by rdiv!
    reshape(result,size(response,1),length(lags),:)
end

function trial_decoder(reg::ProximableFunction,stim,eeg::EEGData,i,lags;
    found_signals=nothing,kwds...)

    stim,response = find_signals(found_signals,stim,eeg,i;kwds...)

    X = withlags(scale(response'),.-reverse(lags))
    Y = view(scale(stim),:,:)

    solver = ProximalAlgorithms.ForwardBackward(fast=true,verbose=true)
    _, A, Y, X = code_init(Val(false),Y,X)
    state = Objective(Y,X,A)
    update!(state,Y,X,0.0)
    solver(state.θ,f=state,g=reg)
    reshape(state.θ,size(response,1),length(lags),:)
end


decode(response::Array,model,lags) =
    withlags(scale(response'),.-reverse(lags)) * reshape(model,:,size(model,3))

function decode_test_cv(train_method,test_method;prefix,indices,group_suffix="",
    name="Training",sources,
    progress=Progress(length(indices)*length(sources),1,desc=name),
    kwds...)

    cachefn(@sprintf("%s_test%s",prefix,group_suffix),
        decode_test_cv_,train_method,test_method;prefix=prefix,
        indices=indices,progress=progress,sources=sources,
        __oncache__ = () ->
            progress_update!(progress,length(indices)*length(sources)),
        kwds...)
end

function single(x)
    @assert(length(x) == 1)
    first(x)
end

function decode_test_cv_(method,::typeof(cor);prefix,eeg,model,lags,indices,stim_fn,
    bounds=all_indices,sources,train_source_indices,progress)

    df = DataFrame()

    for (j,i) in enumerate(indices)
        for (source_index, source) in enumerate(sources)
            train_index = train_source_indices[source_index]
            train_stim = stim_fn(i,train_index)
            train_stim,response = find_signals(nothing,train_stim,eeg,i,
                bounds=bounds[i])

            stim_model = model[train_source_indices[source_index]]
            train_source = sources[train_source_indices[source_index]]
            subj_model_file =
                joinpath(cache_dir(),@sprintf("%s_%s_%02d",train_source,prefix,i))
            # subj_model = load(subj_model_file,"contents")
            subj_model = cachefn(subj_model_file,trial_decoder,
                method,train_stim,eeg,i,lags, bounds = bounds[i],
                found_signals = (train_stim,response))

            test_stim = stim_fn(i,source_index)
            test_stim,response = find_signals(nothing,test_stim,eeg,i,
                bounds=bounds[i])

            # @show subj_model_file

            n = length(indices)
            r1, r2 = (n-1)/n, 1/n

            pred = decode(response,(r1.*stim_model .- r2.*subj_model),
                lags)

            # @show size(pred)
            # @show size(test_stim)
            # @show source
            push!(df,(corr = single(cor(vec(pred),vec(test_stim))),
                source = source, index = j))
            next!(progress)
        end
    end

    categorical!(df,:source)
    df
end
