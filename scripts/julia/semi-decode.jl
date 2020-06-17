using DrWatson
@quickactivate("german_track")
nlags = 17
fs = 64

using EEGCoding, GermanTrack, DataFrames, StatsBase, Underscores, Transducers,
    BangBang, ProgressMeter, Random, Formatting, Serialization, Flux, CUDA

import GermanTrack: stim_info, speakers, directions, target_times, switch_times

traindir = joinpath(cache_dir(),"data")
isdir(traindir) || mkdir(traindir)
trainfile = joinpath(traindir,"semi-decode-train.jlserialize")
if isfile(trainfile)
    data = deserialize(trainfile)
    x,y = data[:x], data[:y]
    weights = data[:weights]
    ii = data[:ii]
else
    eeg_files = dfhit = @_ readdir(processed_datadir()) |>
        filter(occursin(r"^[0-9]+.*\.mcca$",_), __)
    subjects = Dict(
        sidfor(file) => load_subject(joinpath(processed_datadir(), file), stim_info,
                                    encoding = RawEncoding(), framerate = fs)
        for file in eeg_files
    )

    xf = MapSplat((x...) -> find_decoder_training_trials(x...;
        eeg_sr = fs,
        final_sr = fs,
        target_samples = 1fs,
    ))

    segment_definitions = foldl(append!!,xf,
        [(subject,trial) for (_,subject) in subjects for trial in subject.events.trial])
    # segment_definitions.segid = 1:size(segment_definitions,1)

    ntimes = floor(Int,quantile(segment_definitions.len,0.95))

    # break up any segments longer than seglen
    let row = 1
        while row <= size(segment_definitions,1)
            entry = segment_definitions[row,:]
            if entry.len > ntimes
                append!!(segment_definitions,
                    DataFrame(
                        start = entry.start + ntimes,
                        len = entry.len - ntimes;
                        entry[Not([:len,:start])]...
                    ))
                segment_definitions.len[row] = ntimes
            end
            row += 1
        end
    end

    # load the eeg data into memory
    nsegments = size(segment_definitions,1)
    nfeatures = size(first(values(subjects)).eeg[1],1)
    x = Array{Float32}(undef,ntimes,nfeatures*nlags,nsegments);
    let oldtrial = Float64[], oldtriali = 0
        @showprogress "Organizing EEG data..." for (i,segdef) in enumerate(eachrow(segment_definitions))
            # QUESTION: do we do the lags here or inside regress train just before
            # computing the loss?

            # if we're on the same trial, use `oldtrial`, otherwise, load
            # the new data
            trial = if oldtriali != segdef.trial
                oldtriali = segdef.trial
                oldtrial = withlags(subjects[segdef.sid].eeg[segdef.trial]',-(nlags-1):0)
            else
                oldtrial
            end
            start = segdef.start
            stop = min(size(trial,1),segdef.start + segdef.len - 1)
            if stop >= start
                len = stop - start + 1
                x[1:len,:,i] = @view(trial[start:stop,:])
                x[(len+1):end,:,i] .= 0.0
            else
                x[:,:,i] .= 0.0
            end
        end
    end

    # load the envelope and pitch data into memory
    sources = [male_source,fem1_source,fem2_source]
    stim_encoding = JointEncoding(PitchSurpriseEncoding(), ASEnvelope())
    nenc = length(stim_encoding.children)
    nsources = length(sources)
    y = Array{Float32}(undef,ntimes,nenc,nsources,nsegments);
    @showprogress "Organizing Speech data..." for (i,segdef) in enumerate(eachrow(segment_definitions))
        for (h,source) in enumerate(sources)
            event = subjects[segdef.sid].events[segdef.trial,:]
            stim,stim_id = load_stimulus(source,event,stim_encoding,fs,stim_info)
            start = segdef.start
            stop = min(size(stim,1),segdef.start + segdef.len - 1)
            if stop >= start
                len = stop - start + 1
                y[1:len,:,h,i] = @view(stim[start:stop,:])
                y[(len+1):end,:,h,i] .= 0.0
            else
                y[:,:,h,i] .= 0.0
            end
        end
    end


    # define known labels (or weights) for each segment
    hits = @_ segment_definitions |>
        filter(_.target && subjects[_.sid].events[_.trial,:correct],__)
    weights = Array{Float32}(undef,nsources,size(hits,1))
    ii = @_ segment_definitions |> eachrow |>
        findall(_.target && subjects[_.sid].events[_.trial,:correct],__)
    for (i,segdef) in enumerate(eachrow(hits))
        label = subjects[segdef.sid].events[segdef.trial,:target_source]
        if label == 1.0
            weights[:,i] = [1.0,0.0,0.0]
        elseif label == 2.0
            weights[:,i] = [0.0,1.0,0.0]
        else
            error("Unexpected label: $label")
        end
    end

    serialize(trainfile,Dict(
        :x => x, :y => y,
        :weights => weights,
        :ii => ii
    ))
end

# validate the training using a subset of the known labels;
# don't include them as part of the training
N = size(weights,2)
testsize = round(Int,0.2N)
testset = sample(MersenneTwister(1983_11_09), 1:N, testsize, replace=false) |>
    sort!

testweights = weights[:,testset]
testii = ii[testset]
trainweights = weights[:,setdiff(1:end,testset)]
trainii = ii[setdiff(1:end,testset)]

function onvalidate(decoder)
    err = sum((weights(decoder)[testset] .- testweights).^2)
    @info "Test weights have an average error of $(fmt("2.3f",err))"
end

# run decoding
CUDA.allowscalar(false)
result = regressSS2(
    x,y,trainweights,trainii,
    regularize = x -> 0.5f0*sum(abs,x),
    optimizer=AMSGrad(),
    epochs = 1,
    testcb = onvalidate,
)

# TODO: return to older Zygote/CUDA setup, which seemd to work
# in my earlier tests
# store the results
resultfile = joinpath(processed_datadir(),savename("semi-decode-result",
    (nlags=nlags,fs=fs),"bson"))
@tagsave resultfile Dict(
    :weights => weights(result),
    :coefs => EEGCoding.coefs(result),
    :testii => testii
)
