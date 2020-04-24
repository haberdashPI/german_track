using DrWatson
@quickactivate("german_track")

using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures,
    Dates, Underscores, StatsBase, Random, Printf, ProgressMeter, VegaLite,
    FileIO, StatsBase, LIBSVM

using ScikitLearn
@sk_import svm: (NuSVC, SVC)

import GermanTrack: stim_info, speakers, directions, target_times, switch_times

eeg_files = dfhit = @_ readdir(data_dir()) |> filter(occursin(r".mcca$",_), __)
subjects = Dict(file => load_subject(joinpath(data_dir(), file), stim_info,
                                     encoding = RawEncoding())
    for file in eeg_files)


dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

cachefile = joinpath(cache_dir(),"..","data_cache","freqmeans.bson")
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

powerdf = @_ freqmeans |>
    stack(__, Between(:delta,:gamma),
        variable_name = :freqbin, value_name = :power) |>
    filter(all(!isnan,_.power), __)

ε = max(1e-8,minimum(filter(!iszero,powerdf.power))/2)
powerdiff_df = @_ powerdf |>
    unstack(__, :window_timing, :power) |>
    by(__, [:sid,:hit,:freqbin,:condition,:winstart,:winlen,:channel,:salience],
        (:before,:after) => sdf ->
            (powerdiff = mean(log.(ε .+ sdf.after) .- log.(ε .+ sdf.before)),))

powerdiff_df[!,:hit_channel_bin] .=
    categorical(Symbol.(:channel_,powerdiff_df.channel,:_,powerdiff_df.hit,:_,powerdiff_df.freqbin))
classdf = @_ powerdiff_df |>
    unstack(__, [:sid, :condition, :winstart, :winlen, :salience],
        :hit_channel_bin, :powerdiff) |>
    filter(all(!ismissing,_[r"channel"]), __) |>
    disallowmissing!(__,r"channel")

# TODO: optimize nu and gamma
function modelquality(classdf)
    function (row)
        classpredict = by(classdf, [:winstart,:winlen,:salience]) do sdf
            labels = testmodel(NuSVC(nu=row.ν,gamma=row.γ),sdf,:sid,:condition,r"channel")
            DataFrame(correct = sdf.condition .== labels,sid = sdf.sid)
        end
        mean(classpredict.correct)
    end
end

objectdf = @_ classdf |> filter(_.condition in [:global,:object],__)

σ² = var(vec(Array(objectdf[:,r"channel"])))
N = size(objectdf[:,r"channel"],2)
γ_base = 1/(N*σ²)
νs = range(0,0.75,length=9)[2:end]
γs = γ_base * 2.0.^range(-3,3,length=8)
params = DataFrame((ν = ν, γ = γ,) for ν in νs, γ in γs)

paramdir = joinpath(datadir(),"svm_params")
isdir(paramdir) || mkdir(paramdir)
paramfile = joinpath(paramdir,"object_params.csv")

if isfile(paramfile)
    params = CSV.read(paramfile)
else
    params.fitness = @showprogress(map(modelquality(objectdf),eachrow(params)))
    CSV.write(joinpath(datadir(),paramfile),params)
end

pl = params |>
    @vlplot(:rect,
        x={ field=:ν,type=:ordinal },
        y={ field=:γ,type=:ordinal },
        color={:fitness,scale={reverse=true,domain=[0.6,0.7],scheme="plasma"}})

pl |> save(joinpath(dir,"opt-object-params.pdf"))

ν, γ = params[argmax(params.fitness),:]

# TODO: do the same thing, over the full range of sensible values

# TODO: eventually use a validation set not used when plotting
# reporting the results

classpredict = by(objectdf, [:winstart,:winlen,:salience]) do sdf
    labels = testmodel(NuSVC(nu=ν,gamma=γ),sdf,:sid,:condition,r"channel")
    DataFrame(correct = sdf.condition .== labels,sid = sdf.sid)
end

subj_means = @_ classpredict |>
    by(__,[:winstart,:winlen,:salience],:correct => mean)

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
        column=:salience)

save(joinpath(dir,"object_svm_allbins.pdf"),pl)

best_high = @_ subj_means |> filter(_.salience == :high,__) |>
    sort(__,:correct_mean,rev=true) |>
    first(__,1)
best_low = @_ subj_means |> filter(_.salience == :low,__) |>
    sort(__,:correct_mean,rev=true) |>
    first(__,1)

best_vals = @_ classpredict |>
    filter((_1.winstart == best_high.winstart[1] &&
            _1.winlen == best_high.winlen[1]) ||
           (_1.winstart == best_low.winstart[1] &&
            _1.winlen == best_low.winlen[1]),__) |>
    by(__,[:winlen,:salience],:correct => function(x)
        bs = bootstrap(mean,x,BasicSampling(10_000))
        μ,low,high = 100 .* confint(bs,BasicConfInt(0.683))[1]
        (correct = μ, low = low, high = high)
    end)

pl =
    @vlplot() +
    @vlplot(data=[{}], mark=:rule,
    encoding = {
      y = {datum = 50},
      strokeDash = {value = [2,2]}
    }) +
    (best_vals |>
     @vlplot(x={:winlen, type=:ordinal, axis={title="Length (s)"}}) +
     @vlplot(mark={:errorbar,filled=true},
            y={:low,scale={zero=true}, axis={title=""},type=:quantitative},
            y2={:high, type=:quantitative}, color=:salience) +
     @vlplot(mark={:point,filled=true},
            y={:correct,scale={zero=true},axis={title="% Correct Classification"}},
            color=:salience))


# TODO: add a dotted line to chance level

save(joinpath(dir, "object_best_windows.pdf"),pl)

spatialdf = @_ classdf |> filter(_.condition in [:global,:spatial],__)

σ² = var(vec(Array(spatialdf[:,r"channel"])))
N = size(spatialdf[:,r"channel"],2)
γ_base = 1/(N*σ²)
νs = range(0,0.75,length=9)[2:end]
γs = γ_base * 2.0.^range(-3,3,length=8)
params = DataFrame((ν = ν, γ = γ,) for ν in νs, γ in γs)

paramdir = joinpath(datadir(),"svm_params")
isdir(paramdir) || mkdir(paramdir)
paramfile = joinpath(paramdir,"spatial_params.csv")

if isfile(paramfile)
    params = CSV.read(paramfile)
else
    params.fitness = @showprogress(map(modelquality(spatialdf),eachrow(params)))
    CSV.write(joinpath(datadir(),paramfile),params)
end

pl = params |>
    @vlplot(:rect,
        x={ field=:ν,type=:ordinal },
        y={ field=:γ,type=:ordinal },
        color={:fitness,scale={reverse=true,domain=[0.6,0.8],scheme="plasma"}})

pl |> save(joinpath(dir,"opt-spatial-params.pdf"))

ν, γ = params[argmax(params.fitness),:]

# TODO: do the same thing, over the full range of sensible values

# TODO: eventually use a validation set not used when plotting
# reporting the results

classpredict = by(spatialdf, [:winstart,:winlen,:salience]) do sdf
    labels = testmodel(NuSVC(nu=ν,gamma=γ),sdf,:sid,:condition,r"channel")
    DataFrame(correct = sdf.condition .== labels,sid = sdf.sid)
end

subj_means = @_ classpredict |>
    by(__,[:winstart,:winlen,:salience],:correct => mean)

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
        column=:salience)

save(joinpath(dir,"spatial_svm_allbins.pdf"),pl)

best_high = @_ subj_means |> filter(_.salience == :high,__) |>
    sort(__,:correct_mean,rev=true) |>
    first(__,1)
best_low = @_ subj_means |> filter(_.salience == :low,__) |>
    sort(__,:correct_mean,rev=true) |>
    first(__,1)

best_vals = @_ classpredict |>
    filter((_1.winstart == best_high.winstart[1] &&
            _1.winlen == best_high.winlen[1]) ||
           (_1.winstart == best_low.winstart[1] &&
            _1.winlen == best_low.winlen[1]),__) |>
    by(__,[:winlen,:salience],:correct => function(x)
        bs = bootstrap(mean,x,BasicSampling(10_000))
        μ,low,high = 100 .* confint(bs,BasicConfInt(0.683))[1]
        (correct = μ, low = low, high = high)
    end)
best_vals.winlen .= round.(best_vals.winlen,digits=2)

pl =
    @vlplot() +
    @vlplot(data=[{}], mark=:rule,
    encoding = {
      y = {datum = 50, type=:quantitative},
      strokeDash = {value = [2,2]}
    }) +
    (best_vals |>
     @vlplot(x={:winlen, type=:ordinal, axis={title="Length (s)"}}) +
     @vlplot(mark={:errorbar,filled=true},
            y={:low,scale={zero=true}, axis={title=""},type=:quantitative},
            y2={:high, type=:quantitative}, color={:salience, type=:nominal}) +
     @vlplot(mark={:point,filled=true},
            y={:correct,type=:quantitative,scale={zero=true},axis={title="% Correct Classification"}},
            color={:salience, type=:nominal}))

save(File(format"SVG",joinpath(dir, "spatial_best_windows.vegalite")),pl)
