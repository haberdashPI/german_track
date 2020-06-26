using DrWatson
@quickactivate("german_track")

using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures, FFTW,
    Dates, LIBSVM, Underscores, StatsBase, Random, Printf

# eeg_files = filter(x->occursin(r"_mcca03\.mcca_proj$", x), readdir(processed_datadir("eeg")))

eeg_files = filter(x->occursin(r".mcca$", x), readdir(processed_datadir("eeg")))
# eeg_files = filter(x->occursin(r"_cleaned\.eeg$", x), readdir(processed_datadir("eeg")))
eeg_encoding = RawEncoding()

import GermanTrack: stim_info, speakers, directions, target_times, switch_times
subjects = Dict(file => load_subject(joinpath(processed_datadir("eeg"), file), stim_info,
                                     encoding = eeg_encoding)
    for file in eeg_files)

target_salience =
    CSV.read(joinpath(stimulus_dir(), "target_salience.csv")).salience |> Array
med_salience = median(target_salience)

const tindex = Dict("male" => 1, "fem" => 2)

halfway = @_ filter(_ > 0,target_times) |> median
regions = ["target", "baseline"]
timings = ["before", "after"]
winlens = (0.5,1,1.5)
winstarts = (0.0,0.25,0.5)
fs = GermanTrack.framerate(first(values(subjects)).eeg)
factors = Iterators.product(regions,timings,winlens,winstarts)

df = mapreduce(vcat,values(subjects)) do subject
    rows = filter(1:size(subject.events,1)) do i
        !subject.events.bad_trial[i] && subject.events.target_present[i] == 1
    end

    mapreduce(vcat,rows) do row
        si = subject.events.sound_index[row]
        event = subject.events[row,[:correct,:target_present,:target_source,
            :condition,:trial,:sound_index,:target_time]] |> copy

        mapreduce(vcat,factors) do (region,timing,len,start)
            winbounds = timing == "before" ? (-start-len,-start) :
                (start,start+len)
            window = if region == "target"
                only_near(event.target_time,fs,window=winbounds)
            else
                times = vcat(switch_times[si], target_times[si]) |> sort!
                ranges = far_from(times, 10, mindist=0.2, minlength=0.5)
                if isempty(ranges)
                    error("Could not find any valid region for baseline ",
                          "'target'. Times: $(times)")
                end
                at = sample_from_ranges(ranges)
                only_near(at,fs,window=winbounds)
            end

            maxlen = size(subject.eeg[row],2)
            ixs = bound_indices(window,fs,maxlen)
            maxtime = maxlen*fs
            DataFrame(;
                event...,
                region = region,
                timing = timing,
                winlen = len,
                winstart = start,
                sid = subject.sid,
                direction = directions[si],
                salience = target_salience[si] > med_salience ? "high" : "low",
                eeg = [view(subject.eeg[row],:,ixs)],
            )
        end
    end
end
source_names = ["male", "female"]
df.target_source = get.(Ref(source_names),Int.(df.target_source),missing)
str2sym(x) = x
str2sym(x::String) = Symbol(x)
df = str2sym.(df)
categorical!(df)

df.hit = ishit.(eachrow(df))
dfhit = df[in.(df.hit,Ref((:hit,:miss,:baseline))),:]

freqbins = OrderedDict(
    :delta => (1,3),
    :theta => (3,7),
    :alpha => (7,15),
    :beta => (15,30),
    :gamma => (30,100),
)

fs = GermanTrack.framerate(first(values(subjects)).eeg)
# channels = first(values(subjects)).eeg.label
channels = 1:30
function freqrange(spect,(from,to))
    freqs = range(0,fs/2,length=size(spect,2))
    view(spect,:,findall(from-step(freqs)*0.51 .≤ freqs .≤ to+step(freqs)*0.51))
end

freqmeans = by(dfhit, [:sid,:hit,:timing,:condition,:winstart,:winlen,:salience]) do rows
    # @assert size(rows,1) == 1
    # signal = rows.eeg[1]
    signal = reduce(hcat,row.eeg for row in eachrow(rows))
    if size(signal,2) < 100
        empty = mapreduce(hcat,keys(freqbins)) do bin
            DataFrame(Symbol(bin) => Float64[])
        end
        empty[!,:channel] = Int[]
        return empty
    end
    spect = abs.(rfft(signal, 2))
    # totalpower = mean(spect,dims = 2)
    result = mapreduce(hcat,keys(freqbins)) do bin
        mfreq = mean(freqrange(spect, freqbins[bin]), dims = 2) #./ totalpower
        DataFrame(Symbol(bin) => vec(mfreq))
    end
    result[!,:channel] .= channels
    result
end

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

svmdf = @_ freqmeans |>
    filter((_1.winstart == 0.25 && _1.winlen == 0.5) ||
           (_1.winstart == 0.5 && _1.winlen == 1.5),__) |>
    filter(_.condition in [:global,:object],__) |>
    stack(__, [:delta,:theta,:alpha,:beta,:gamma],
        variable_name = :freqbin, value_name = :power) |>
    unstack(__, :timing, :power) |>
    by(__, [:sid,:freqbin,:condition,:winstart,:winlen,:channel,:salience],
        (:before,:after) => sdf ->
            (powerdiff = mean(log.(sdf.after) .- log.(sdf.before)),))

function svmaccuracy(sdf,cols)
    N = 0
    correct = 0
    for (train_ids,test_ids) in folds(10,unique(sdf.sid))
        train = @_ filter(_.sid in train_ids,sdf)
        test = @_ filter(_.sid in test_ids,sdf)
        model = svmtrain(Array(disallowmissing(train[:,cols]))',
                         Array(train[:,:condition]))
        labels, = svmpredict(model, Array(disallowmissing(test[:, cols]))')
        @infiltrate
        N += size(test,1)
        correct += sum(labels .== test[:, :condition])
    end
    DataFrame(N=N,correct=correct)
end

svmclass = @_ by(svmaccuracy(_,[:powerdiff]),svmdf,
    [:winstart,:winlen,:channel,:salience,:freqbin])
svmclass[!,:channelgroup] = @_ map(@sprintf("channel%02d",_),svmclass.channel)

rnd = MersenneTwister(1983)
rseqs = [sort!(sample(rnd,1:30,5,replace=false)) for _ in 1:10]
channel_groups = OrderedDict(
    "1-5" => 1:5,
    "1-10" => 1:10,
    "1-20" => 1:20,
    (join(r,",") => r for r in rseqs)...
)

for group in keys(channel_groups)
    channels = channel_groups[group]
    newrows = by(svmdf,[:winstart,:winlen,:salience,:freqbin]) do sdf
        @_ sdf |>
            filter(_.channel in channels,__) |>
            unstack(__,:channel,:powerdiff,renamecols=Symbol(:channel,_)) |>
            svmaccuracy(__,All(r"channel[0-9]+"))
    end
    newrows[!,:channelgroup] .= group
    newrows[!,:channel] .= maximum(svmclass.channel)+1
    append!(svmclass,newrows)
end

sym2str(x) = x
sym2str(x::Symbol) = string.(x)
svmclass = sym2str.(svmclass)

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

R"""

library(ggplot2)
library(dplyr)
library(tidyr)

bins = $(collect(keys(freqbins)))

plotdf = $svmclass %>%
    mutate(freqbin = factor(freqbin,levels=bins, ordered=T)) %>%
    arrange(freqbin)

p = ggplot(plotdf,aes(x=channel,y=freqbin,fill=correct/N)) +
    geom_raster() + facet_grid(winlen~salience,labeller="label_both") +
    scale_fill_distiller(name="Label Accuracy (global v object)",
        na.value="gray95",palette="PuBuGn",limits=c(0.5,0.75),direction=0) +
    scale_x_continuous(breaks=c(0,10,20,30,
        31:$(30+length(keys(channel_groups)))),
        labels=c(0,10,20,30,$(collect(keys(channel_groups))))) +
    xlab('MCCA Component Grouping') +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))

p

ggsave(file.path($dir,"svm_freqbin_salience.pdf"),plot=p,width=11,height=6)

sortdf = plotdf %>% group_by(salience,freqbin,winlen) %>%
    arrange(desc(correct/N)) %>%
    mutate(rank = row_number())

p = ggplot(sortdf, aes(x=rank, y=correct/N, color=freqbin)) +
    geom_line() + scale_color_brewer(palette='RdYlGn') +
    facet_grid(winlen~salience,labeller="label_both") +
    geom_abline(intercept=0.5,slope=0,linetype=2) +
    xlab('Rank of MCCA Component Grouping')

p

ggsave(file.path($dir,"svm_freqbin_channel_rank_salience.pdf"),plot=p,width=11,height=6)

"""

svmdf = @_ freqmeans |>
    filter((_1.winstart == 0.25 && _1.winlen == 0.5) ||
           (_1.winstart == 0.5 && _1.winlen == 1.5),__) |>
    filter(_.condition in ["global","spatial"],__) |>
    stack(__, [:delta,:theta,:alpha,:beta,:gamma],
        variable_name = :freqbin, value_name = :power) |>
    unstack(__, :timing, :power) |>
    by(__, [:sid,:freqbin,:condition,:winstart,:winlen,:channel,:salience],
        (:before,:after) => sdf ->
            (powerdiff = mean(log.(sdf.after) .- log.(sdf.before)),))

svmclass = @_ by(svmaccuracy(_,[:powerdiff]),svmdf,
    [:winstart,:winlen,:channel,:salience,:freqbin])
svmclass[!,:channelgroup] = @_ map(@sprintf("channel%02d",_),svmclass.channel)

rnd = MersenneTwister(1983)
rseqs = [sort!(sample(rnd,1:30,5,replace=false)) for _ in 1:10]
channel_groups = OrderedDict(
    "1-5" => 1:5,
    "1-10" => 1:10,
    "1-20" => 1:20,
    (join(r,",") => r for r in rseqs)...
)

for group in keys(channel_groups)
    channels = channel_groups[group]
    newrows = by(svmdf,[:winstart,:winlen,:salience,:freqbin]) do sdf
        @_ sdf |>
            filter(_.channel in channels,__) |>
            unstack(__,:channel,:powerdiff,renamecols=Symbol(:channel,_)) |>
            svmaccuracy(__,All(r"channel[0-9]+"))
    end
    newrows[!,:channelgroup] .= group
    newrows[!,:channel] .= maximum(svmclass.channel)+1
    append!(svmclass,newrows)
end

svmclass.freqbin = String.(svmclass.freqbin)

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

R"""

library(ggplot2)
library(dplyr)
library(tidyr)

bins = $(collect(keys(freqbins)))

plotdf = $svmclass %>%
    mutate(freqbin = factor(freqbin,levels=bins, ordered=T)) %>%
    arrange(freqbin)

p = ggplot(plotdf,aes(x=channel,y=freqbin,fill=correct/N)) +
    geom_raster() + facet_grid(winlen~salience,labeller="label_both") +
    scale_fill_distiller(name="Label Accuracy (global v object)",
        na.value="gray95",palette="PuBuGn",limits=c(0.5,0.75),direction=0) +
    scale_x_continuous(breaks=c(0,10,20,30,
        31:$(30+length(keys(channel_groups)))),
        labels=c(0,10,20,30,$(collect(keys(channel_groups))))) +
    xlab('MCCA Component Grouping') +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))

p

ggsave(file.path($dir,"spatial_svm_freqbin_salience.pdf"),plot=p,width=11,height=6)

sortdf = plotdf %>% group_by(salience,freqbin,winlen) %>%
    arrange(desc(correct/N)) %>%
    mutate(rank = row_number())

p = ggplot(sortdf, aes(x=rank, y=correct/N, color=freqbin)) +
    geom_line() + scale_color_brewer(palette='RdYlGn') +
    facet_grid(winlen~salience,labeller="label_both") +
    geom_abline(intercept=0.5,slope=0,linetype=2) +
    xlab('Rank of MCCA Component Grouping')

p

ggsave(file.path($dir,"spatial_svm_freqbin_channel_rank_salience.pdf"),plot=p,width=11,height=6)

"""
