
# TODO: why is high classification worse, early on
# try to reproduce old plot


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

# powerdiff[!,:hit_channel] .=
#     categorical(Symbol.(map(x -> @sprintf("channel%02d",x),powerdiff.channel),
#         :_,powerdiff.hit))

powerdiff[!,:hit_channel] .=
    categorical(Symbol.(map(x -> @sprintf("channel%02d",x),powerdiff.channel)))

classdf = @_ powerdiff |>
    unstack(__, [:sid, :freqbin, :condition, :winstart, :winlen, :salience],
        :hit_channel, :powerdiff) |>
    filter(all(!ismissing,_[r"channel"]), __) |>
    disallowmissing!(__,r"channel")

classpredict = by(classdf, [:freqbin,:winstart,:winlen,:salience]) do sdf
    mapreduce(vcat,1:30) do channel
        labels = testmodel(LIBSVM.SVC(),sdf,:sid,:condition,Regex(@sprintf("channel%02d",channel)))
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
        labels = testmodel(LIBSVM.SVC(),sdf,:sid,:condition,cols)
        DataFrame(correct = sdf.condition .== labels,sid = sdf.sid)
    end
    newrows[!,:channelgroup] .= group
    newrows[!,:channel] .= maximum(classpredict.channel)+1
    append!(classpredict,newrows)
end
classpredict.correct = Int.(classpredict.correct)

classpredict |>
    @vlplot(:rect,
        x={:channel,type=:quantitative,bin={step=1}},
        y={:freqbin,type=:ordinal,
           sort=reverse([:delta,:theta,:alpha,:beta,:gamma])},
        column=:salience,
        color={"mean(correct)",type=:quantitative,scale={reverse=true,domain=[0.5,1],scheme="plasma"}},
        row=:winlen) #, column=:salience,row=:winlen)

mean_correct = by(classpredict,[:freqbin,:salience,:winlen,:channel],:correct => mean)
mean_correct.salience = string.(mean_correct.salience)
mean_correct.freqbin = string.(mean_correct.freqbin)
R"""

ggplot($mean_correct,aes(x=channel,y=freqbin,fill=correct_mean)) +
    geom_raster() + facet_grid(winlen~salience)
# TODO: try SVC, rather than NuSVC, try LIBSVM
# then try collapsing across hits, misses and baselines
"""
