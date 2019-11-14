using DrWatson; quickactivate(@__DIR__, "german_track")
include(joinpath(srcdir(), "julia", "setup.jl"))

stim_info = JSON.parsefile(joinpath(stimulus_dir(), "config.json"))
eeg_files = filter(x->occursin(r"_mcca34\.mcca_proj$", x), readdir(data_dir()))
# eeg_files = filter(x->occursin(r"_cleaned\.eeg$", x), readdir(data_dir()))
eeg_encoding = RawEncoding()

subjects = Dict(file =>
    load_subject(joinpath(data_dir(), file),
        stim_info,
        encoding = eeg_encoding)
    for file in eeg_files)

const speakers = convert(Array{Int},
    stim_info["test_block_cfg"]["trial_target_speakers"])
const tindex = Dict("male" => 1, "fem" => 2)
const cond_label = Dict("object" => "object",
    "global" => "test",
    "spatial" => "feature")
const direction = convert(Array{String},
    stim_info["test_block_cfg"]["trial_target_dir"])

target_times =
    convert(Array{Float64}, stim_info["test_block_cfg"]["target_times"])

before_window(time,start,len) =
    iszero(time) ? no_indices : only_near(time,10, window=(-start-len,-start))
after_window(time,start,len) =
    iszero(time) ? no_indices : only_near(time,10, window=(start,start+len))

windows = Dict(
    "before" => before_window,
    "after" => after_window
)

conditions = Dict((
    sid = sid,
    label = label,
    timing = timing,
    condition = condition,
    winstart = start,
    winlen = len,
    target = target
) => @λ(_row.condition == cond_label[condition] &&
        ((label == "detected") == _row.correct) &&
        sid == _row.sid &&
        speakers[_row.sound_index] == tindex[target] ?
            windows[timing](target_times[_row.sound_index],start,len) :
                no_indices)
    for condition in keys(cond_label)
    for target in keys(tindex)
    for label in ["detected", "not_detected"]
    for timing = keys(windows)
    for start in (0,0.25,0.5)
    for len in (0.5,1,1.5)
    for sid in sidfor.(eeg_files)
)

df = DataFrame()
freqs = nothing
times = nothing
for (condition, bounds) in conditions
    global df

    bounds = Dict((file, i) => bounds
        for file in eeg_files
        for (i, bounds) in enumerate(apply_bounds(bounds, subjects[file])))
    indices = filter(@λ(!isempty(bounds[_])), keys(bounds)) |> collect |> sort!

    if isempty(indices)
        error("No valid bounds for condition: $condition")
    end
    for (file, i) in indices
        eeg, events = subjects[file]
        start = bounds[(file,i)][1]
        ixs = bound_indices(bounds[(file, i)], 256, size(eeg[i], 2))

        # power in relevant frequency bins across all channels and times
        df = push!(df, (
            sid = sidfor(file),
            trial = i,
            condition...,
            window = view(eeg[i],:,ixs)
        ))
    end
end

freqbins = OrderedDict(
    "delta" => (1,3),
    "theta" => (3,7),
    "alpha" => (7,15),
    "beta" => (15,30),
    "gamma" => (30,100),
)

fs = GermanTrack.samplerate(first(values(subjects)).eeg)
# channels = first(values(subjects)).eeg.label
channels = 1:34
function freqrange(spect,(from,to))
    freqs = range(0,fs/2,length=size(spect,2))
    view(spect,:,findall(from .≤ freqs .≤ to))
end

freqmeans = by(df, [:sid,:label,:timing,:condition,:target,:winstart,:winlen]) do rows
    signal = reduce(hcat,row.window for row in eachrow(rows))
    spect = abs.(fft(signal, 2))
    # totalpower = mean(spect,dims = 2)

    result = mapreduce(hcat,keys(freqbins)) do bin
        result = DataFrame()
        mfreq = mean(freqrange(spect, freqbins[bin]), dims = 2) #./ totalpower
        DataFrame(Symbol(bin) => vec(mfreq))
    end
    result[!,:channel] .= channels
    result
end

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

R"""
library(ggplot2)
library(dplyr)
library(tidyr)

bins = $(collect(keys(freqbins)))

df = $(freqmeans) %>%
    group_by(sid,winstart,winlen,label,timing,condition,target) %>%
    gather(key="freqbin", value="meanpower", delta:gamma) %>%
    filter(sid != 11) %>%
    ungroup() %>%
    mutate(timing = factor(timing,levels=c('before','after')),
           freqbin = factor(freqbin,levels=bins, ordered=T)) %>%
    arrange(timing,freqbin)

df = filter(df,sid != 11)

group_means = df %>%
    group_by(sid,winstart,winlen,label,timing,condition,target,freqbin) %>%
    summarize(meanpower = median(meanpower))

for(start in unique(df$winstart)){
    for(len in unique(df$winlen)){
        plotdf = filter(group_means,winstart == start,winlen == len) %>%
            group_by(sid,label,condition,target) %>%
            spread(timing,meanpower) %>%
            mutate(diff = after - before)

        pos = position_jitter(width=0.1)
        p = ggplot(plotdf,aes(x=label,y=diff)) +
            stat_summary(geom='bar',position=position_dodge(width=1),
                aes(fill=label),size=4) +
            stat_summary(geom='linerange',position=position_dodge(width=1)) +
            geom_point(alpha=0.5,color='black', position=pos) +
            scale_fill_brewer(palette='Set1',direction=-1) +
            scale_color_brewer(palette='Set1',direction=-1) +
            facet_grid(freqbin~condition+target,scales='free_y') +
            ylab("Median power difference across channels (after - before)")
        name = sprintf('freq_diff_summary_with_mcca_%03.1f_%03.1f.pdf',start,len)
        ggsave(file.path($dir,name),plot=p,width=11,height=8)
    }
}

"""
