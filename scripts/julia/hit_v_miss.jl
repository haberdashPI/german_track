using DrWatson; @quickactivate("german_track")
include(joinpath(srcdir(), "julia", "setup.jl"))

stim_info = JSON.parsefile(joinpath(stimulus_dir(), "config.json"))
# eeg_files = filter(x->occursin(r"_mcca03\.mcca_proj$", x), readdir(data_dir()))
eeg_files = filter(x->occursin(r"_mcca34\.mcca_proj$", x), readdir(data_dir()))
# eeg_files = filter(x->occursin(r"_cleaned\.eeg$", x), readdir(data_dir()))
eeg_encoding = RawEncoding()

subjects =
    Dict(file => load_subject(joinpath(data_dir(), file), stim_info,
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
switch_times =
    convert(Array{Array{Float64}},stim_info["test_block_cfg"]["switch_times"])
fs = convert(Float64,stim_info["fs"])
switch_times = map(times -> times ./ fs,switch_times)

halfway = filter(@λ(_ > 0),target_times) |> median

before_window(time,start,len) =
    iszero(time) ? no_indices : only_near(time,10, window=(-start-len,-start))
after_window(time,start,len) =
    iszero(time) ? no_indices : only_near(time,10, window=(start,start+len))

windows = Dict(
    "before" => before_window,
    "after" => after_window
)

baseline_len = 0.5
conditions = Dict((
    sid = sid,
    label = label,
    timing = timing,
    condition = condition,
    winstart = start,
    winlen = len,
    direction = dir,
    target = target) =>

    function(row)
        if (row.condition == cond_label[condition] &&
            ((label == "detected") == row.correct) &&
            sid == row.sid &&
            direction[row.sound_index] == dir &&
            (target == "baseline" ||
             speakers[row.sound_index] == tindex[target]))

            if target == "baseline"
                times = vcat(switch_times[row.sound_index],
                    target_times[row.sound_index]) |> sort!
                ranges = far_from(times,10, mindist=0.2,minlength=baseline_len)
                if isempty(ranges)
                    error("Could not find any valid region for baseline ",
                          "'target'. Times: $(times)")
                end
                windows[timing](sample_from_ranges(ranges), 0, baseline_len)
            else
                windows[timing](target_times[row.sound_index], start, len)
            end
        else
            no_indices
        end
    end

    for condition in keys(cond_label)
    for target in vcat(collect(keys(tindex)),"baseline")
    for label in ["detected", "not_detected"]
    for dir in unique(direction)
    for timing in keys(windows)
    # for target_timing in ["early", "late"]
    for start in (0.0,0.25,0.5)
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

    if !isempty(indices)
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
end

function ishit(row)
    if row.condition == "global"
        row.target == "baseline" ? "baseline" :
            row.label == "detected" ? "hit" : "miss"
    elseif row.condition == "object"
        row.target == "baseline" ? "baseline" :
            row.target == "male" ?
                (row.label == "detected" ? "hit" : "miss") :
                (row.label == "detected" ? "falsep" : "reject")
    else
        @assert row.condition == "spatial"
        row.target == "baseline" ? "baseline" :
            row.direction == "right" ?
                (row.label == "detected" ? "hit" : "miss") :
                (row.label == "detected" ? "falsep" : "reject")
    end
end

df.hit = ishit.(eachrow(df))
dfhit = df[in.(df.hit,Ref(("hit","miss","baseline"))),:]


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

freqmeans = by(dfhit, [:sid,:hit,:timing,:condition,##:target_timing,
    :winstart,:winlen]) do rows

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

# TODO: show the varying window parameters within a single graph
# so I can understand how things change,
# also, think about the comparisons I care about
# and other ways to collapse across more of the figures (e.g. colors
# for the different frequency bands, or something)

df = $(freqmeans) %>%
    filter(channel %in% 1:3) %>%
    group_by(sid,winstart,winlen,hit,#target_timing,
    timing,condition) %>%
    gather(key="freqbin", value="meanpower", delta:gamma) %>%
    filter(sid != 11) %>%
    ungroup() %>%
    mutate(timing = factor(timing,levels=c('before','after')),
           freqbin = factor(freqbin,levels=bins, ordered=T)) %>%
    arrange(timing,freqbin)

df = filter(df,sid != 11)

group_means = df %>%
    group_by(sid,winstart,winlen,hit,#target_timing,
    timing,condition,freqbin) %>%
    summarize(meanpower = median(meanpower))

# for(start in unique(df$winstart)){
#     for(len in unique(df$winlen)){
plotdf = filter(group_means,freqbin %in% c('delta','theta','alpha')) %>%
    # filter(condition %in% c('global','object')) %>%
    #filter(group_means,winstart == start,winlen == len) %>%
    filter(hit %in% c('hit','miss')) %>%
    group_by(sid,hit,condition) %>%
    spread(timing,meanpower) %>%
    mutate(diff = after - before)

pos = position_jitterdodge(dodge.width=0.1,jitter.width=0.05)

p = ggplot(plotdf,aes(x=winstart,y=diff,
    fill=interaction(hit,factor(winlen)),
    color=interaction(hit,factor(winlen)))) +
    stat_summary(geom='line',position=position_dodge(width=0.1),
        size=1) +
    stat_summary(geom='linerange',position=position_dodge(width=0.1)) +
    geom_point(alpha=0.1, position=pos) +
    scale_fill_brewer(palette='Paired',direction=-1) +
    scale_color_brewer(palette='Paired',direction=-1) +
    facet_grid(freqbin~condition,scales='free_y') +
    ylab("Median power difference across channels (after - before)") +
    # coord_cartesian(ylim=c(-0.006,0.006))
    geom_abline(slope=0,intercept=0,linetype=2)
p

# name = sprintf('freq_diff_summary_target_timing_%03.1f_%03.1f.pdf',start,len)
# name = sprintf('mcca_freq_diff_summary_target_timing_%03.1f_%03.1f.pdf',start,len)
name = 'mcca_hits_all_windows.pdf'
ggsave(file.path($dir,name),plot=p,width=11,height=8)
#     }
# }

plotdf = group_means %>%
    filter(winstart == 0.0,winlen %in% c(0.5,1.5)) %>%
    group_by(sid,hit,condition) %>%
    spread(timing,meanpower) %>%
    mutate(diff = after - before)

pos = position_jitterdodge(dodge.width=0.1,jitter.width=0.05)

p = ggplot(plotdf,aes(x=freqbin,y=diff,
        fill=hit,color=hit)) +
    stat_summary(fun.data = "mean_cl_boot", geom='point',
        position=position_dodge(width=0.1),
        size=1,fun.args = list(conf.int=0.68)) +
    stat_summary(fun.data = "mean_cl_boot", geom='errorbar',
        position=position_dodge(width=0.1), width=0.5) +
    geom_point(alpha=0.4, position=pos, size=1) +
    # geom_text(position=pos, size=4, aes(label=sid)) +
    scale_fill_brewer(palette='Set1',direction=-1) +
    scale_color_brewer(palette='Set1',direction=-1) +
    facet_grid(winlen~condition,scales='free_y') +
    ylab("Median power difference across channels (after - before)") +
    geom_abline(slope=0,intercept=0,linetype=2) +
    coord_cartesian(ylim=c(-0.002,0.002))
p

name = sprintf('mcca03_hits_%03.1f_%03.1f.pdf',0.25,1)
ggsave(file.path($dir,name),plot=p,width=11,height=8)

"""
