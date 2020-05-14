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

trial_definitions = reduce(append!!,xf,
    [(subject,trial) for (_,subject) in subjects for trial in subject.events.trial])

# TODO: figure out how big the data set is, is it big enough to fit in memory?
# TODO: build up the actual data set and then run the actual problem
