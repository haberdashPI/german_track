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
target_window_range = (-1, 1)
target_window = map(target_times) do time
    iszero(time) ? no_indices :
        only_near(time, 10, window = target_window_range)
end

conditions = Dict(
    (sid = sid, label = label, condition = condition, target = target) =>
        @λ(_row.condition == cond_label[condition] &&
           ((label == "correct") == _row.correct) &&
           sid == _row.sid &&
           speakers[_row.sound_index] == tindex[target] ?
                target_window[_row.sound_index] : no_indices)
    for condition in keys(cond_label)
    for target in keys(tindex)
    for label in ["correct", "incorrect"]
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
    μ = padmeanpower(rows.alphapower)
    curtimes = range(target_window_range[1], length = size(μ, 2), step = 1 / fs)
    alpha = PlotAxes.asplotable(AxisArray(μ,
        Axis{:channel}(Base.axes(μ, 1)),
        Axis{:time}(curtimes),
    ), quantize = (1000, 1000))[1]
    rename!(alpha, :value => :alphapower)

    μ = padmeanpower(rows.gammapower)
    curtimes = range(target_window_range[1], length = size(μ, 2), step = 1 / fs)
    gamma = PlotAxes.asplotable(AxisArray(μ,
        Axis{:channel}(Base.axes(μ, 1)),
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
        Axis{:time}(curtimes),
    ), quantize = (1000, 1000))[1]
end

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

R"""
library(ggplot2)
library(dplyr)

ggplot($freqmeans,aes(x=time,y=channel,color=value)) + geom_raster() +
    facet_grid(sid+target~condition+label)

df = $(freqmeans) %>% filter(label == 'correct')
df = df %>%
    mutate(case = ifelse(target == 'male',
        label,ifelse(label == 'correct','false_alarm (female)','ignore'))) %>%
    filter(case != 'ignore')

ggplot(df,aes(x=time,y=channel,fill=alphaz)) + geom_raster() +
    facet_grid(case~sid+condition) +
    scale_fill_distiller(palette='RdBu')

ggsave(file.path($dir,'alpha_ind.pdf'),width=11,height=8)

p = ggplot(df,aes(x=time,y=channel,fill=alphaz)) + geom_raster() +
    facet_grid(case~condition) +
    scale_fill_distiller(palette='RdBu')

ggsave(file.path($dir,'alpha_mean.pdf'),plot=p,width=11,height=8)

p = ggplot(df,aes(x=time,y=channel,fill=gammaz)) + geom_raster() +
    facet_grid(case~sid+condition) +
    scale_fill_distiller(palette='RdBu')

ggsave(file.path($dir,'gamma_ind.pdf'),plot=p,width=11,height=8)

p = ggplot(df,aes(x=time,y=channel,fill=gammaz)) + geom_raster() +
    facet_grid(case~condition) +
    scale_fill_distiller(palette='RdBu')

ggsave(file.path($dir,'gamma_mean.pdf'),plot=p,width=11,height=8)

p = ggplot(df,aes(x=time,y=alphaz,color=condition)) + stat_summary(geom='line') +
    facet_grid(case~.)
ggsave(file.path($dir,'alpha_time.pdf'),plot=p,width=11,height=8)

p = ggplot(df,aes(x=time,y=gammaz,color=condition)) + stat_summary(geom='line') +
    facet_grid(case~.)
ggsave(file.path($dir,'gamma_time.pdf'),plot=p,width=11,height=8)

# ggplot(df,aes(x=channel,y=gammaz,color=condition)) + stat_summary(geom='line') +
#     coord_flip() +
#     facet_grid(target~.)

# df = $(spectmeans) %>% filter(label == 'correct',sid == 8,freq > 14)

# ggplot(df,aes(x=time,y=freq,fill=log(value))) + geom_raster() +
#     facet_grid(target~condition) +
#     scale_fill_distiller(palette='Spectral')

# df = $(spectmeans) %>% filter(sid == 8,freq > 14)

# ggplot(df,aes(x=time,y=freq,fill=log(value))) + geom_raster() +
#     facet_grid(target+label~condition) +
#     scale_fill_distiller(palette='Spectral')

# df = $(spectmeans) %>% filter(freq > 14)

# ggplot(df,aes(x=time,y=freq,fill=log(value))) + geom_raster() +
#     facet_grid(target+label~sid+condition) +
#     scale_fill_distiller(palette='Spectral')

"""

# TODO: organize by 'correct', 'incorrect' for male target and
# by distractor 'female'
