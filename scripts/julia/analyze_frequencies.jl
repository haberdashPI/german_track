using DrWatson; quickactivate(@__DIR__, "german_track"); using GermanTrack
using DSP
using StatsBase
using PaddedViews

stim_info = JSON.parsefile(joinpath(stimulus_dir(), "config.json"))
eeg_files = filter(x->occursin(r"_mcca34\.mcca_proj$", x), readdir(data_dir()))
eeg_encoding = JointEncoding(RawEncoding(),
    FilteredPower("alpha", 5, 15),
    FilteredPower("gamma", 30, 100))

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
target_window_range = (-1.5, 2)
target_window = map(target_times) do time
    iszero(time) ? no_indices :
        only_near(time, 10, window = target_window_range)
end

conditions = Dict(
    (sid = sid, label = label, condition = condition, target = target) =>
        @λ(_row.condition == cond_label[condition] &&
           ((label == "correct") == _row.correct) &&
           speakers[_row.sound_index] == tindex[target] ?
                target_window[_row.sound_index] : no_indices)
    for condition in keys(cond_label)
    for target in keys(tindex)
    for label in ["correct", "incorrect"]
    for sid in sidfor.(eeg_files)
)

# a few options
# - show alpha power for each individual feature
# - show mean power spectrogram of all features

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
        ixs = bound_indices(bounds[(file, i)], 256, size(eeg[i], 2))

        # power in relevant frequency bins across all channels and times
        alphapower = @views(eeg[i][35:68,ixs])
        gammapower = @views(eeg[i][69:end,ixs])

        # power over all frequencies and times (averaged across channels)
        mspect = mean(1:34) do ch
            data = @views(eeg[i][ch,ixs])
            spect = spectrogram(data, 128, 112, # 500ms window with 87.5% overlap
                fs = 256,
                window = DSP.Windows.hanning)
            global freqs = freq(spect)
            global times = time(spect)
            power(spect)
        end

        df = push!(df, (sid = sidfor(file),
            trial = i,
            condition...,
            spectrum = mspect,
            alphapower = alphapower,
            gammapower = gammapower,
        ))
    end
end

fs = GermanTrack.samplerate(first(values(subjects)).eeg)
freqmeans = by(df, [:sid,:label,:condition,:target]) do rows
    row1 = rows[1,:]
    curtimes = range(0, length = size(row1.alphapower, 2), step = 1 / fs)
    alpha = PlotAxes.asplotable(AxisArray(mean(rows.alphapower),
        Axis{:channel}(Base.axes(row1.alphapower, 1)),
        Axis{:time}(curtimes),
    ), quantize = (1000, 1000))[1]
    rename!(alpha, :value => :alphapower)

    gamma = PlotAxes.asplotable(AxisArray(mean(rows.alphapower),
        Axis{:channel}(Base.axes(row1.gammapower, 1)),
        Axis{:time}(curtimes),
    ), quantize = (1000, 1000))[1]
    rename!(gamma,:value => :gammapower)

    result = join(alpha,gamma,on=[:channel,:time])
    result[!,:alphaz] .= zscore(result.alphapower)
    result[!,:gammaz] .= zscore(result.gammapower)
    result
end

spectmeans = by(df, [:sid,:label,:condition,:target]) do rows
    maxtime = maximum(@λ(size(_,2)), rows.spectrum)
    nf = size(rows.spectrum[1],1)

    curtimes = range(
        target_window_range[1]+step(times),
        step=step(times),
        length=maxtime
    )

    # power is always postive so we treat -1 as a missing value
    # and compute the mean over non-missing values; `PaddedView`
    # does not support `missing` values.
    padded = map(@λ(PaddedView(-1,_spect,(nf,maxtime))),rows.spectrum)
    μ = zeros(nf,maxtime)
    for pad in padded
        μ .= ifelse.(pad .>= 0,μ .+ pad,μ)
    end

    PlotAxes.asplotable(AxisArray(μ,
        Axis{:freq}(freqs),
        Axis{:time}(times),
    ), quantize = (1000, 1000))[1]
end

R"""
library(ggplot2)
library(dplyr)

df = $(means) %>% filter(label == 'correct',sid == 8)

ggplot($means,aes(x=time,y=channel,color=value)) + geom_raster() +
    facet_grid(sid+target~condition+label)

df = $(freqmeans) %>% filter(label == 'correct',sid == 8)

ggplot(df,aes(x=time,y=channel,fill=alphaz)) + geom_raster() +
    facet_grid(target~condition) +
    scale_fill_distiller(palette='RdBu')

ggplot(df,aes(x=time,y=channel,fill=gammaz)) + geom_raster() +
    facet_grid(target~condition) +
    scale_fill_distiller(palette='RdBu')

ggplot(df,aes(x=time,y=zscore)) + stat_summary(geom='line') +
    facet_grid(target~condition)

ggplot(df,aes(x=channel,y=zscore)) + stat_summary(geom='line') +
    coord_flip() +
    facet_grid(target~condition)

df = $(spectmeans) %>% filter(label == 'correct',sid == 8,freq > 14)

ggplot(df,aes(x=time,y=freq,fill=log(value))) + geom_raster() +
    facet_grid(target~condition) +
    scale_fill_distiller(palette='Spectral')

df = $(spectmeans) %>% filter(sid == 8,freq > 14)

ggplot(df,aes(x=time,y=freq,fill=log(value))) + geom_raster() +
    facet_grid(target+label~condition) +
    scale_fill_distiller(palette='Spectral')

df = $(spectmeans) %>% filter(freq > 14)

ggplot(df,aes(x=time,y=freq,fill=log(value))) + geom_raster() +
    facet_grid(target+label~condition+sid) +
    scale_fill_distiller(palette='Spectral')

"""

# TODO: organize by 'correct', 'incorrect' for male target and
# by distractor 'female'
