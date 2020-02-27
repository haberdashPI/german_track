using DrWatson; @quickactivate("german_track")
include(joinpath(srcdir(), "julia", "setup.jl"))

# eeg_files = filter(x->occursin(r"_mcca03\.mcca_proj$", x), readdir(data_dir()))
eeg_files = filter(x->occursin(r"_mcca34\.mcca_proj$", x), readdir(data_dir()))
# eeg_files = filter(x->occursin(r"_cleaned\.eeg$", x), readdir(data_dir()))

fs = 32
eeg_encoding = FFTFiltered("delta" => (1.0,3.0),seconds=15,fs=fs,nchannels=34)
stim_encoding = JointEncoding(PitchSurpriseEncoding(), ASEnvelope())
sources = [male_source,fem1_source,fem2_source,mixed_sources,
           fem_mix_sources,joint_source,other(male_source)]

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

const tindex = Dict("male" => 1, "fem" => 2)

during_target = map(target_times) do time
    iszero(time) ? no_indices : only_near(time,10,window=(0,1.5))
end

listen_conds = ["object","global"]
sids = getproperty.(values(subjects),:sid)
condition_targets = Dict("object" => [1], "global" => [1,2])

conditions = Dict(
    (label=label,condition=condition) =>
        @λ(_row.condition == condition &&
           (label == "all" || _row.correct) &&
           (speakers[_row.sound_index] ∈ condition_targets[condition]) ?
                during_target[_row.sound_index] : no_indices)
    for condition in listen_conds
    for label in ["correct", "all"]
)

df = DataFrame()
df[!,:label] = Union{Int,Missing}[]

# TODO: using the simpler approach of collecting on the data transparently
# should get rid of a lot of old code; and serve as a double check on any bugs

# the plan is to first look at the indices that are actually
# being trained and tested vs. the folds
N = sum(@λ(_subj.events,1), values(subjects))
progress = Progress(N,desc="Assembling Data: ")
for subject in values(subjects)
    rows = filter(1:size(subject.events,1)) do i
        !subject.events.bad_trial[i]
    end

    for row in 1:size(subject.events,1)
        si = subject.events.sound_index[row]
        event = subject.events[row,[:correct,:target_present,:target_source,
            :condition,:trial,:sound_index,:target_time]] |> copy

        window = only_near(target_times[si],fs,window=(0,0.5))

        for source in sources
            stim, = load_stimulus(source,event,stim_encoding,fs,stim_info)
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

        next!(progress)
    end
end

df = train_test(
    StaticMethod(NormL2(0.2),cor),
    SpeakerStimMethod(
        encoding=encoding,
        sources=),
    subjects = subjects,
    encode_eeg = eeg_encoding,
    resample = 10,
    eeg_files, stim_info,
    maxlag=0.8,
    train = subdict(conditions,
        (label = "correct", condition = cond) for cond in listen_conds
    ),
    test = subdict(conditions,
        (label = "all", condition = cond) for cond in listen_conds
    )
);
alert()

# df[!,:location] = directions[df.stim_id]

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

R"""

library(tidyr)
library(dplyr)
library(ggplot2)
library(stringr)

ggplot($df,aes(x=source,y=value,color=test_correct)) +
    stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
        position=position_nudge(x=0.3)) +
    geom_point(alpha=0.5,position=position_jitter(width=0.1)) +
    scale_color_brewer(palette='Set1') +
    facet_grid(train_condition~sid,labeller=label_context)

group_mean = $(df) %>%
    mutate(source = factor(source,c('male','other_male','fem1','fem2',
        'all','fem1+fem2','joint'))) %>%
    group_by(sid,source,test_correct,train_condition) %>%
    summarize(value=mean(value))

ggplot($df,aes(x=source,y=value,color=test_correct)) +
    stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
        position=position_nudge(x=0.3)) +
    geom_point(alpha=0.5,position=position_jitter(width=0.1)) +
    geom_abline(intercept=0,slope=0,linetype=2) +
    scale_color_brewer(palette='Set1') +
    facet_grid(train_condition~.,labeller=label_context)

ggsave(file.path($dir,"during_target_delta_decode.pdf"),width=11,height=8)

"""
