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

function find_decoder_training_trials(subject,trial)
    result = Empty(DataFrame)
    si = subject.events.sound_index[trial]

    # the labeled segment
    target_time = target_times[si]
    condition = subject.events.condition[trial]
    target_source = subject.events.target_source
    if target_time > 0
        weights =
            if condition == "global"
                [1/3 1/3 1/3]
            elseif condition == "object"
                [1.0 0.0 0.0]
            elseif condition == "spatial"
                if target_source == 1.0
                    [1.0 0.0 0.0]
                elseif target_source == 2.0
                    [0.0 1.0 0.0]
                else
                    error("Unexpected target source: ",
                        target_source)
                end
            else
                error("Unexpected condition: ",condition)
            end

        append!!(result, DataFrame(
            weights = weights,
            start = round(Int,target_time*final_sample_rate),
            len = target_samples,
        ))
    end

    si = subject.events.sound_index[trial]
    n = size(subject.eeg[trial],2)
    for (start,stop) in far_from([target_times[si]; switch_times[si]],n/eeg_sr)
        start_sample = round(Int,start*final_sample_rate)
        append!!(result, DataFrame(
            weights = missing,
            start = start_sample,
            len = min(n,round(Int,stop * final_sample_rate) - start_sample + 1)
        ))
    end
end

# WIP: get these definitions to work
trial_definitions = reduce(append!!,MapSplat(find_decoder_training_trials),
    [(subject,trial) for (_,subject) in subjects for trial in subject.events.trial])

# TODO: figure out how big the data set is, is it big enough to fit in memory?
# TODO: build up the actual data set and then run the actual problem
