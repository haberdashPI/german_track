using DrWatson
@quickactivate("german_track")

using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures,
    Dates, Underscores, StatsBase, Random, Printf, ProgressMeter, VegaLite,
    FileIO, StatsBase, LIBSVM

sample = StatsBase.sample

using ScikitLearn
@sk_import svm: (NuSVC, SVC)

import GermanTrack: stim_info, speakers, directions, target_times, switch_times

eeg_files = dfhit = @_ readdir(processed_datadir()) |> filter(occursin(r".mcca$",_), __)
subjects = Dict(file => load_subject(joinpath(processed_datadir(), file), stim_info,
                                     encoding = RawEncoding())
    for file in eeg_files)

cachefile = joinpath(cache_dir(),"data","freqmeans.bson")
if !isfile(cachefile)
    freqmeans = organize_data_by(
        subjects,groups=[:salience],hittypes = [:hit,:miss,:baseline],
        winlens = 2.0 .^ range(-3,1,length=10),
        winstarts = 2.0 .^ range(-3,1,length=10)) do signal,fs
            result = computebands(signal,fs)
            if @_ all(0 ≈ _,signal)
                result[:,Between(:delta,:gamma)] .= 0
            end
            result
        end
    @save cachefile freqmeans
    alert()
else
    @load cachefile freqmeans
end

powerdiff = @_ freqmeans |>
    filter((isapprox(_1.winstart,0.23,atol=0.02) &&
            isapprox(_1.winlen,0.58,atol=0.02)) ||
           (isapprox(_1.winstart,0.58,atol=0.02) &&
            isapprox(_1.winlen,1.46,atol=0.02)),__) |>
    filter(_.condition in [:global,:object],__) |>
    stack(__, [:delta,:theta,:alpha,:beta,:gamma],
        variable_name = :freqbin, value_name = :power) |>
    unstack(__, :window_timing, :power) |>
    by(__, [:sid,:hit,:freqbin,:condition,:winstart,:winlen,:channel,:salience],
        (:before,:after) => sdf ->
            (powerdiff = mean(log.(sdf.after) .- log.(sdf.before)),))

powerdiff[!,:hit_channel] .=
    categorical(Symbol.(map(x -> @sprintf("channel%02d",x),powerdiff.channel),
        :_,powerdiff.hit))

# powerdiff[!,:hit_channel] .=
#     categorical(Symbol.(map(x -> @sprintf("channel%02d",x),powerdiff.channel)))

classdf = @_ powerdiff |>
    unstack(__, [:sid, :freqbin, :condition, :winstart, :winlen, :salience],
        :hit_channel, :powerdiff) |>
    filter(all(!ismissing,_[r"channel"]), __) |>
    disallowmissing!(__,r"channel")

classpredict = by(classdf, [:freqbin,:winstart,:winlen,:salience]) do sdf
    mapreduce(vcat,1:30) do channel
        labels = testclassifier(NuSVC(),sdf,:sid,:condition,Regex(@sprintf("channel%02d",channel)))
        DataFrame(correct = sdf.condition .== labels,sid = sdf.sid,channel = channel)
    end
end

classpredict[!,:channelgroup] = @_ map(@sprintf("channel%02d",_),classpredict.channel)

rnd = MersenneTwister(1983)
rseqs = [sort!(sample(rnd,1:30,5,replace=false)) for _ in 1:10]
channel_groups = OrderedDict(
    "1-5" => 1:5,
    "1-10" => 1:10,
    "1-20" => 1:20,
    "all" => 1:30,
    (join(r,",") => r for r in rseqs)...
)

for group in keys(channel_groups)
    chs = channel_groups[group]
    newrows = by(classdf,[:freqbin,:winstart,:winlen,:salience]) do sdf
        cols = Regex("("*join((@sprintf("channel%02d",ch) for ch in chs),"|")*")")
        labels = testclassifier(NuSVC(),sdf,:sid,:condition,cols)
        DataFrame(correct = sdf.condition .== labels,sid = sdf.sid)
    end
    newrows[!,:channelgroup] .= group
    newrows[!,:channel] .= maximum(classpredict.channel)+1
    append!(classpredict,newrows)
end
classpredict.correct = Int.(classpredict.correct)
class_means = by(classpredict,[:freqbin,:channel,:salience,:winlen],:correct => mean)

pl = class_means |>
    @vlplot(:rect,
        x={:channel,type=:quantitative,bin={step=1}},
        y={:freqbin,type=:ordinal,
           sort=reverse([:delta,:theta,:alpha,:beta,:gamma])},
        column={:salience, typ=:ordinal},
        color={:correct_mean ,type=:quantitative,
               scale={reverse=true,domain=[0.5,1],scheme="plasma"}},
        row={:winlen, typ=:ordinal}) #, column=:salience,row=:winlen)

mean_correct = by(classpredict,[:freqbin,:salience,:winlen,:channel],:correct => mean)
mean_correct.salience = string.(mean_correct.salience)
mean_correct.freqbin = string.(mean_correct.freqbin)

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

save(File(format"PDF",joinpath(dir,"susvm_allbins.pdf")),pl)

