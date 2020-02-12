using DrWatson; @quickactivate("german_track")
include(joinpath(srcdir(), "julia", "setup.jl"))

# eeg_files = filter(x->occursin(r"_mcca03\.mcca_proj$", x), readdir(data_dir()))
eeg_files = filter(x->occursin(r"_mcca34\.mcca_proj$", x), readdir(data_dir()))
# eeg_files = filter(x->occursin(r"_cleaned\.eeg$", x), readdir(data_dir()))
eeg_encoding = RawEncoding()

import GermanTrack: stim_info, speakers, directions, target_times, switch_times
subjects = Dict(file => load_subject(joinpath(data_dir(), file), stim_info,
                                     encoding = eeg_encoding)
    for file in eeg_files)

const tindex = Dict("male" => 1, "fem" => 2)

halfway = filter(@λ(_ > 0),target_times) |> median

windows = Dict(
    "before" => (time,start,len) -> iszero(time) ? no_indices :
        only_near(time,10, window=(-start-len,-start)),
    "after" => (time,start,len) -> iszero(time) ? no_indices :
        only_near(time,10, window=(start,start+len))
)

conditions = Dict((
    sid = sid,
    label = label,
    timing = timing,
    condition = condition,
    direction = dir,
    target = target) =>

    function(row)
        if (row.condition == condition &&
            ((label == "detected") == row.correct) &&
            sid == row.sid &&
            directions[row.sound_index] == dir &&
            (target == "baseline" ||
             speakers[row.sound_index] == tindex[target]))

            if target == "baseline"
                times = vcat(switch_times[row.sound_index],
                    target_times[row.sound_index]) |> sort!
                ranges = far_from(times,10, mindist=0.2,minlength=0.5)
                if isempty(ranges)
                    error("Could not find any valid region for baseline ",
                          "'target'. Times: $(times)")
                end
                windows[timing](sample_from_ranges(ranges), 0, 1.5)
            else
                windows[timing](target_times[row.sound_index], 0.0, 1.5)
            end
        else
            no_indices
        end
    end

    for condition in unique(first(subjects)[2].events.condition)
    for target in vcat(collect(keys(tindex)),"baseline")
    for label in ["detected", "not_detected"]
    for dir in unique(directions)
    for timing in keys(windows)
    # for target_timing in ["early", "late"]
    for sid in getproperty.(values(subjects),:sid)
)
df = select_windows(conditions, subjects)

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


fs = GermanTrack.samplerate(first(values(subjects)).eeg)
# channels = first(values(subjects)).eeg.label
channels = 1:34

using DSP: Periodograms
freqmeans = by(dfhit, [:sid,:trial,:hit,:timing,:condition]) do rows
    signal = reduce(hcat,row.window for row in eachrow(rows))
    if size(signal,2) < 100
        DataFrame(power = Float64[], channel = Int[], timebin = Int[],
            freqbin = Int[])
    end
    spects = mapreduce(vcat,Base.axes(signal,1)) do ch
        spect = abs.(stft(signal[ch,:], 32, fs=256, window=DSP.Windows.hanning))
        reshape(spect,1,size(spect)...)
    end

    # totalpower = mean(spect,dims = 2)

    DataFrame(power = vec(spects),
        channel = vec(getindex.(CartesianIndices(spects),1)),
        timebin = vec(getindex.(CartesianIndices(spects),2)),
        freqbin = vec(getindex.(CartesianIndices(spects),3)))
end

using Feather
Feather.write("timefreq_windows.feather",freqmeans)

# TODO: plot median time-freq diff across conditions
# use below to load old data, rather than re-running above analysis
freqmeans = Feather.read("timefreq_windows.feather")
