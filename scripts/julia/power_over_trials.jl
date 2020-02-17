
using DrWatson; @quickactivate("german_track")
include(joinpath(srcdir(), "julia", "setup.jl"))

stim_info = JSON.parsefile(joinpath(stimulus_dir(), "config.json"))
# eeg_files = filter(x->occursin(r"_mcca03\.mcca_proj$", x), readdir(data_dir()))
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

timewindow(time,start,len) =
    iszero(time) ? no_indices :

conditions = Dict((time = time,sid = sid,trial = trial) =>
    row -> sid == row.sid ? only_near(time,10,window=(0,0.5)) : no_indices
    for time in range(0,10,step=0.5)
    for sid in sidfor.(eeg_files)
    for trial in 1:150
)

df = DataFrame()
freqs = nothing
times = nothing
using Threads
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

freqbins = OrderedDict(
    "delta" => (1,3),
    "theta" => (3,7),
    "alpha" => (7,15),
    "beta" => (15,30),
    "gamma" => (30,100),
)

fs = GermanTrack.framerate(first(values(subjects)).eeg)
# channels = first(values(subjects)).eeg.label
channels = 1:34
function freqrange(spect,(from,to))
    freqs = range(0,fs/2,length=size(spect,2))
    view(spect,:,findall(from .≤ freqs .≤ to))
end

freqmeans = by(df, :trial) do rows
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
    group_by(trial) %>%
    gather(key="freqbin", value="meanpower", delta:gamma) %>%
    # filter(sid != 11) %>%
    ungroup() %>%
    mutate(freqbin = factor(freqbin,levels=bins, ordered=T)) %>%
    arrange(trial,freqbin)

group_means = df %>%
    group_by(trial,freqbin) %>%
    summarize(meanpower = median(meanpower))

p = ggplot(group_means,aes(x=trial,y=meanpower,color=freqbin)) + geom_line()

name = 'mean_power_by_trial.pdf'
ggsave(file.path($dir,name),plot=p,width=7,height=5)


"""
