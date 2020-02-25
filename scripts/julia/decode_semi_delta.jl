using DrWatson; @quickactivate("german_track")
include(joinpath(srcdir(), "julia", "setup.jl"))

# eeg_files = filter(x->occursin(r"_mcca03\.mcca_proj$", x), readdir(data_dir()))
eeg_files = filter(x->occursin(r"_mcca34\.mcca_proj$", x), readdir(data_dir()))
# eeg_files = filter(x->occursin(r"_cleaned\.eeg$", x), readdir(data_dir()))

fs = 32
eeg_encoding = FFTFiltered("delta" => (1.0,3.0),seconds=15,fs=fs,nchannels=34)
encoding = JointEncoding(PitchSurpriseEncoding(), ASEnvelope())

import GermanTrack: stim_info, speakers, directions, target_times, switch_times

cachefile = joinpath(cache_dir(),"..","subject_cache","delta_subjects$(fs).bson")
if isfile(cachefile)
    @load cachefile subjects
else
    subjects = Dict(file =>
        load_subject(joinpath(data_dir(), file),
            stim_info,
            encoding = eeg_encoding,
            framerate=fs)
        for file in eeg_files)
    @save cachefile subjects
end

function target_label(row)::Union{Missing,Int}
    if row.correct && row.target_present
        if row.condition == "global"
            row.target_source
        elseif row.condition == "object" && row.target_source == 1
            1
        else
            missing
        end
    else
        missing
    end
end

source_indices = ["male", "fem"]
conditions = ["object", "global"]
df = DataFrame(
    correct=Bool[],
    target_present=Bool[],
    target_source=Int[],
    condition=String[],
    window=AbstractArray{Float64,2}[],
    label=Union{Int,Missing}[],
    sid=Int[],
)
for subject in values(subjects)
    rows = filter(1:size(subject.events,1)) do i
        subject.events.condition[i] in conditions &&
        !subject.events.bad_trial[i]
    end

    for row in 1:size(subject.events,2)
        si = subject.events.sound_index[row]
        event = subject.events[row,[:correct,:target_present,:target_source,
            :condition]] |> copy

        windows = vcat(
            DataFrame(
                range = only_near(target_times[si],fs,window=(0,1)),
                hastarget = true
            ),
            DataFrame(
                range = not_near([target_times[si]; switch_times[si]],
                    fs,window=(0,0.5)),
                hastarget = false
            )
        )

        for window in eachrow(windows)
            ixs = bound_indices(window.range,fs,size(subject.eeg[row],2))
            push!(df,merge(event,(
                window = view(subject.eeg[row],:,ixs),
                label = window.hastarget ? target_label(event) : missing,
                sid = subject.sid
            )))
        end
    end
end

models = by(df, [:condition,:sid]) do cond
    stim =
