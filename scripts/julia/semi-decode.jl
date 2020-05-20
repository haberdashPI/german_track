using DrWatson
@quickactivate("german_track")

using EEGCoding, GermanTrack, DataFrames, StatsBase, Underscores, Transducers,
    BangBang

import GermanTrack: stim_info, speakers, directions, target_times, switch_times

fs = 64
eeg_files = dfhit = @_ readdir(data_dir()) |> filter(occursin(r".mcca$",_), __)
subjects = Dict(
    sidfor(file) => load_subject(joinpath(data_dir(), file), stim_info,
                                 encoding = RawEncoding(), framerate = fs)
    for file in eeg_files
)

xf = MapSplat((x...) -> find_decoder_training_trials(x...;
    eeg_sr = fs,
    final_sr = fs,
    target_samples = target_samples
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
x = Array{Float64}(undef,ntimes,nfeatures,nsegments);
for (i,segdef) in enumerate(eachrow(segment_definitions))
    trial = subjects[segdef.sid].eeg[segdef.trial]
    start = segdef.start
    stop = min(size(trial,2),segdef.start + segdef.len - 1)
    if stop > start
        len = stop - start + 1
        x[1:len,:,i] = @view(trial[:,start:stop])'
        x[(len+1):end,:,i] .= 0.0
    else
        x[:,:,i] .= 0.0
    end
end

# load the envelope and pitch data into memory
sources = [male_source,fem1_source,fem2_source]
nenv = 2
nsources = length(sources)
stim_encoding = JointEncoding(PitchSurpriseEncoding(), ASEnvelope())
y = Array{Float64}(undef,ntimes,nenv,nsources,nsegments)
for (i,segdef) in enumerate(eachrow(segment_definitions))
    for (h,source) in enumerate(sources)
        event = subjects[segdef.sid].events[segdef.trial,:]
        stim,stim_id = load_stimulus(source,event,stim_encoding,fs,stim_info)
        start = segdef.start
        stop = min(size(stim,2),segdef.start + segdef.len - 1)
        if stop > start
            len = stop - start + 1
            y[1:len,:,h,i] = @view(stim[start:stop,:])
            y[(len+1):end,:,h,i] .= 0.0
        else
            y[:,:,h,i] .= 0.0
        end
    end
end

# TODO: setup the labels
# remember that we need to know that the segment has a target and that
# the segment is on a correct trial to give the segment known weights

# run the training
