using DrWatson
@quickactivate("german_track")
include(joinpath("..", "..", "src", "julia", "setup.jl"))

# eeg_files = filter(x->occursin(r"_mcca03\.mcca_proj$", x), readdir(data_dir()))
eeg_files = filter(x->occursin(r"_mcca34\.mcca_proj$", x), readdir(data_dir()))
# eeg_files = filter(x->occursin(r"_cleaned\.eeg$", x), readdir(data_dir()))

fs = 32
eeg_encoding = FFTFiltered("delta" => (1.0,3.0),seconds=15,fs=fs,nchannels=34)
stim_encoding = JointEncoding(PitchSurpriseEncoding(), ASEnvelope())
nlags = round(Int,0.5*fs)
conditions = ["global", "object"]
sources = [JointSource(false), other(JointSource(false))]

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

if !isdefined(Main,:df)
    N = sum(@λ(size(_subj.events,1)), values(subjects))
    progress = Progress(N,desc="Assembling Data: ")
    df = mapreduce(vcat,values(subjects)) do subject
        rows = filter(1:size(subject.events,1)) do i
            subject.events.condition[i] in conditions &&
            !subject.events.bad_trial[i]
        end

        mapreduce(vcat,1:size(subject.events,1)) do row
            si = subject.events.sound_index[row]
            event = subject.events[row,[:correct,:target_present,:target_source,
                :condition,:trial,:sound_index,:target_time]] |> copy

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
            next!(progress)

            mapreduce(vcat,eachrow(windows)) do window
                for source in sources
                    stim, = load_stimulus(source,event,stim_encoding,fs,stim_info)
                    stim = mapslices(slice -> withlags(slice,0:nlags),stim,dims=(2,3))
                    maxlen = min(size(subject.eeg[row],2),size(stim,2))
                    ixs = bound_indices(window.range,fs,maxlen)
                    DataFrame(;event...,
                        eeg = [view(subject.eeg[row],:,ixs)],
                        stim = [permutedims(view(stim,:,ixs,:),(1,3,2))],
                        source = string(source),
                        label = window.hastarget ? target_label(event) : missing,
                        sid = subject.sid
                    )
                end
            end
        end
    end
end

models = by(df, [:source]) do sdf
    labeled = findall(.!ismissing.(sdf.label))
    k = size(sdf.stim[1],1)
    labels = isempty(labeled) ? [] : reduce(hcat,onehot.(skipmissing(sdf.label),k))'

    model = regressSS(sdf.eeg,sdf.stim,labels,labeled,CvNorm(0.2,1);
        max_iter = 10_0000)
    (model = model,)
end

