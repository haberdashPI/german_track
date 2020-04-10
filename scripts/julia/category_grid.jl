using DrWatson
@quickactivate("german_track")

using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures,
    Dates, Underscores, StatsBase, Random, Printf, ProgressMeter, VegaLite,
    FileIO

using ScikitLearn
@sk_import svm: NuSVC

import GermanTrack: stim_info, speakers, directions, target_times, switch_times

eeg_files = dfhit = @_ readdir(data_dir()) |> filter(occursin(r".mcca$",_), __)
subjects = Dict(file => load_subject(joinpath(data_dir(), file), stim_info,
                                     encoding = RawEncoding())
    for file in eeg_files)

freqmeans = organize_freqbands(subjects,groups=[:salience],hittypes = [:hit,:miss,:baseline],
    winlens = 2.0 .^ range(-3,1,length=10),
    winstarts = 2.0 .^ range(-3,1,length=10))
alert()

powerdf = @_ freqmeans |>
    filter(_.condition in [:global,:object],__) |>
    stack(__, Between(:delta,:gamma),
        variable_name = :freqbin, value_name = :power) |>
    filter(all(!isnan,_.power), __)

ε = max(1e-8,minimum(filter(!iszero,powerdf.power))/2)
powerdiff_df = @_ powerdf |>
    unstack(__, :window_timing, :power) |>
    by(__, [:sid,:hit,:freqbin,:condition,:winstart,:winlen,:channel,:salience],
        (:before,:after) => sdf ->
            (powerdiff = mean(log.(ε .+ sdf.after) .- log.(ε .+ sdf.before)),))

powerdiff_df[!,:hit_channel] .= categorical(Symbol.(:channel_,powerdiff_df.channel,:_,powerdiff_df.hit))
classdf = @_ powerdiff_df |>
    unstack(__, [:sid, :freqbin, :condition, :winstart, :winlen, :salience],
        :hit_channel, :powerdiff) |>
    filter(all(!ismissing,_[r"channel"]), __) |>
    disallowmissing!(__,r"channel")


# TODO: for some reason we still have infinite values in powerdiff
# figure that out and then create the plot
classpredict = by(classdf, [:freqbin,:winstart,:winlen,:salience]) do sdf
    labels = testmodel(NuSVC(),sdf,:sid,:condition,r"channel")
    DataFrame(correct = sdf.condition .== labels,sid = sdf.sid)
end

subj_means = @_ classpredict |>
    by(__,[:winstart,:winlen,:freqbin,:salience],:correct => mean)

sort!(subj_means,order(:correct_mean,rev=true))
first(subj_means,6)

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

subj_means.llen = log.(2,subj_means.winlen)
subj_means.lstart = log.(2,subj_means.winstart)

pl = subj_means |>
    @vlplot(:rect,
        x={
            field=:lstart,
            bin={step=4/9,anchor=-3-2/9},
        },
        y={
            field=:llen,
            bin={step=4/9,anchor=-3-2/9},
        },
        color={:correct_mean,scale={reverse=true,domain=[0.5,1],scheme="plasma"}},
        column=:salience,
        row={field=:freqbin,type=:ordinal,
             sort=[:delta,:theta,:alpha,:beta,:gamma]})

save(File(format"PDF",joinpath(dir,"svm_allbins.pdf")),pl)

highvlow = @_ subj_means |>
    unstack(__,:salience,:correct_mean) |>
    by(__, [:winstart,:winlen,:freqbin],
        (:low,:high) => row -> (diff = row.low - row.high,))

highvlow.llen = log.(2,highvlow.winlen)
highvlow.lstart = log.(2,highvlow.winstart)

pl = highvlow |>
    @vlplot(:rect,
        x={
            field=:lstart,
            bin={step=4/9,anchor=-3-2/9},
        },
        y={
            field=:llen,
            bin={step=4/9,anchor=-3-2/9},
        },
        color={field=:diff,scale={domain=[-0.5,0.5], scheme="redblue"}},
        row={field=:freqbin,type=:ordinal,
             sort=[:delta,:theta,:alpha,:beta,:gamma]})

save(File(format"PDF",joinpath(dir,"diff_svm_allbins.pdf")),pl)
