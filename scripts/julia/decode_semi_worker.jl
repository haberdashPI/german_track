import GermanTrack: stim_info, speakers, directions, target_times, switch_times

fs = 32
cachefile = joinpath(cache_dir(),"eeg","delta_subjects$(fs).bson")
@load cachefile subjects

stim_encoding = JointEncoding(PitchSurpriseEncoding(), ASEnvelope())
nlags = round(Int,0.5*fs)
conditions = ["global", "object"]
sources = [JointSource(false), other(JointSource(false))]

function target_label(row)
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

df = DataFrame(
    correct=Bool[],
    target_present=Bool[],
    target_source=Int[],
    condition=String[],
    trial=Int[],
    sound_index=Int[],
    target_time=Float64[],
    eeg=AbstractArray{Float64,2}[],
    stim=AbstractArray{Float64,3}[],
    source=String[],
    label=Union{Int,Missing}[],
    sid=Int[],
)

N = sum(@λ(size(_subj.events,1)), values(subjects))
progress = Progress(N,desc="Assembling Data: ")
for subject in values(subjects)
    rows = filter(1:size(subject.events,1)) do i
        subject.events.condition[i] in conditions &&
        !subject.events.bad_trial[i]
    end

    for row in 1:size(subject.events,1)
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

        for window in eachrow(windows)
            for source in sources
                stim, = load_stimulus(source,event,stim_encoding,fs,stim_info)
                # todo: the EEG, not the stimulus should be lagged (right??)
                stim = mapslices(slice -> withlags(slice,0:nlags),stim,dims=(2,3))
                maxlen = min(size(subject.eeg[row],2),size(stim,2))
                ixs = bound_indices(window.range,fs,maxlen)
                push!(df,merge(event,(
                    eeg = view(subject.eeg[row],:,ixs),
                    stim = permutedims(view(stim,:,ixs,:),(1,3,2)),
                    source = string(source),
                    label = window.hastarget ? target_label(event) : missing,
                    sid = subject.sid
                )))
            end
        end
    end
end

function findmodel(sdf)
    labeled = findall(.!ismissing.(sdf.label))
    k = size(sdf.stim[1],1)
    labels = isempty(labeled) ? [] :
        reduce(hcat,onehot.(skipmissing(sdf.label),k))'

    model = regressSS(sdf.eeg,sdf.stim,labels,labeled,CvNorm(0.2,1);
        max_iter = 10_0000)
    (model = model,)
end


