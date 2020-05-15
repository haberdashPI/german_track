using DrWatson
@quickactivate("german_track")

using EEGCoding, GermanTrack, DataFrames, StatsBase, Underscores, Transducers,
    BangBang

import GermanTrack: stim_info, speakers, directions, target_times, switch_times

eeg_files = dfhit = @_ readdir(data_dir()) |> filter(occursin(r".mcca$",_), __)
subjects = Dict(file => load_subject(joinpath(data_dir(), file), stim_info,
                                    encoding = RawEncoding())
    for file in eeg_files)

const eeg_sr = GermanTrack.framerate(first(values(subjects)).eeg)

const final_sample_rate = 64
const target_duration = 1
const target_samples = 64

# WIP: get these definitions to work
xf = MapSplat((x...) -> find_decoder_training_trials(x...;
    eeg_sr = eeg_sr,
    final_sr = final_sample_rate,
    target_samples = target_samples
))

trial_definitions = foldl(append!!,xf,
    [(subject,trial) for (_,subject) in subjects for trial in subject.events.trial])

seglen = floor(Int,quantile(trial_definitions.len,0.95))

# break up any segments longer than len
for row in 1:size(trial_definitions,1)
    entry = trial_definitions[row,:]
    if entry.len > seglen
        push!!(trial_definitions,
            DataFrame(start = entry.start + seglen,))

# TODO: load the eeg data into memory
x = Array{Float64}()

# TODO: load the evenlop and pitch data into memory

# run the training
