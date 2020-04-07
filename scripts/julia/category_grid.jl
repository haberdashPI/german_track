using DrWatson
@quickactivate("german_track")

using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures,
    Dates, Underscores, StatsBase, Random, Printf, ProgressMeter, VegaLite

using ScikitLearn
@sk_import svm: NuSVC

import GermanTrack: stim_info, speakers, directions, target_times, switch_times

eeg_files = dfhit = @_ readdir(data_dir()) |> filter(occursin(r".mcca$",_), __)
subjects = Dict(file => load_subject(joinpath(data_dir(), file), stim_info,
                                     encoding = RawEncoding())
    for file in eeg_files)

freqmeans = organize_freqbands(subjects,groups=[:salience],
    winlens=range(0,2,length=10),winstarts=range(0,2,length=10))

classdf = @_ freqmeans |>
    filter(_.condition in [:global,:object],__) |>
    stack(__, Between(:delta,:gamma),
        variable_name = :freqbin, value_name = :power) |>
    filter(all(!isnan,_.power), __)

ε = max(1e-8,minimum(filter(!iszero,classdf.power))/2)
classdf = @_ classdf |>
    unstack(__, :window_timing, :power) |>
    by(__, [:sid,:freqbin,:condition,:winstart,:winlen,:channel,:salience],
        (:before,:after) => sdf ->
            (powerdiff = mean(log.(ε .+ sdf.after) .- log.(ε .+ sdf.before)),))

classdf_shape = @_ classdf |>
    unstack(__, [:sid, :freqbin, :condition, :winstart, :winlen, :salience],
        :channel, :powerdiff, renamecols = Symbol(:channel,_)) |>
    filter(all(!ismissing,_[r"channel"]), __) |>
    disallowmissing!(__,r"channel")


# TODO: for some reason we still have infinite values in powerdiff
# figure that out and then create the plot
classpredict = by(classdf_shape, [:freqbin,:winstart,:winlen,:salience]) do sdf
    labels = testmodel(NuSVC(),sdf,:sid,:condition,r"channel")
    DataFrame(correct = sdf.condition .== labels,sid = sdf.sid)
end

subj_means = @_ classpredict |>
    by(__,[:winstart,:winlen,:freqbin,:salience],:correct => mean)

sort!(subj_means,order(:correct_mean,rev=true))
first(subj_means,6)

subj_means |>
    @vlplot(:rect, x=:winstart,y=:winlen,color="mean(correct_mean)",column=:salience)
