using DrWatson
@quickactivate("german_track")
include(joinpath(srcdir(), "julia", "setup.jl"))

# eeg_files = filter(x->occursin(r"_mcca03\.mcca_proj$", x), readdir(data_dir()))
eeg_files = filter(x->occursin(r"_mcca34\.mcca_proj$", x), readdir(data_dir()))
# eeg_files = filter(x->occursin(r"_cleaned\.eeg$", x), readdir(data_dir()))

fs = 32
eeg_encoding = FFTFiltered("delta" => (1.0,3.0),seconds=15,fs=fs,nchannels=34)
stim_encoding = JointEncoding(PitchSurpriseEncoding(), ASEnvelope())
sources = [male_source,fem1_source,fem2_source,mixed_sources,
           fem_mix_sources,joint_source,other(male_source)]
nlags = round(Int,fs*0.25)

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

df = DataFrame(
    correct=Bool[],
    target_present=Bool[],
    target_source=Int[],
    condition=String[],
    trial=Int[],
    sound_index=Int[],
    target_time=Float64[],
    eeg=AbstractArray{Float64,2}[],
    stim=AbstractArray{Float64,2}[],
    source=String[],
    sid=Int[],
)

# TODO: using the simpler approach of collecting on the data transparently
# should get rid of a lot of old code; and serve as a double check on any bugs

# the plan is to first look at the indices that are actually
# being trained and tested vs. the folds
N = sum(@λ(size(_subj.events,1)), values(subjects))
progress = Progress(N,desc="Assembling Data: ")
for subject in values(subjects)
    rows = filter(1:size(subject.events,1)) do i
        !subject.events.bad_trial[i]
    end

    for row in 1:size(subject.events,1)
        si = subject.events.sound_index[row]
        event = subject.events[row,[:correct,:target_present,:target_source,
            :condition,:trial,:sound_index,:target_time]] |> copy

        window = only_near(event.target_time,fs,window=(0,0.5))

        for source in sources
            stim, = load_stimulus(source,event,stim_encoding,fs,stim_info)
            maxlen = min(size(subject.eeg[row],2),size(stim,2))
            ixs = bound_indices(window,fs,maxlen)
            push!(df,merge(event,(
                eeg = view(subject.eeg[row]',ixs,:),
                stim = view(stim,ixs,:),
                source = string(source),
                sid = subject.sid
            )))
        end

        next!(progress)
    end
end

# santity check, train all data in each condition, and then test all
# data (then implement k-fold validation)
dfcorr = df[(df.target_present .== true) .& (df.correct .== true),:]
N = groupby(dfcorr, [:condition,:source,:target_source]) |> size()
progress = Progress(N,desc="Building decoders: ")
models = by(dfcorr, [:condition,:source,:target_source]) do sdf
    model = decoder(NormL2(0.2),reduce(vcat,sdf.stim),
        reduce(vcat,sdf.eeg), 0:nlags)
    next!(progress)
    (model = model,)
end

dftest = df[(df.target_present .== true)]
results = by(dftest,[:condition,:source,:target_source]) do sdf
    smodels = models[
        (models.condition .== sdf.condition[1]) .&
        (models.source .== sdf.source[1]) .&
        (models.target_source .== sdf.target_source[1]), :]
    @assert size(smodels,1) == 1
    model = smodels.model[1]
    by(sdf,[:trial]) do trial

    end
end




# df = train_test(
#     StaticMethod(NormL2(0.2),cor),
#     SpeakerStimMethod(
#         encoding=encoding,
#         sources=),
#     subjects = subjects,
#     encode_eeg = eeg_encoding,
#     resample = 10,
#     eeg_files, stim_info,
#     maxlag=0.8,
#     train = subdict(conditions,
#         (label = "correct", condition = cond) for cond in listen_conds
#     ),
#     test = subdict(conditions,
#         (label = "all", condition = cond) for cond in listen_conds
#     )
# );
# alert()

# df[!,:location] = directions[df.stim_id]

# dir = joinpath(plotsdir(),string("results_",Date(now())))
# isdir(dir) || mkdir(dir)

# R"""

# library(tidyr)
# library(dplyr)
# library(ggplot2)
# library(stringr)

# ggplot($df,aes(x=source,y=value,color=test_correct)) +
#     stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
#         position=position_nudge(x=0.3)) +
#     geom_point(alpha=0.5,position=position_jitter(width=0.1)) +
#     scale_color_brewer(palette='Set1') +
#     facet_grid(train_condition~sid,labeller=label_context)

# group_mean = $(df) %>%
#     mutate(source = factor(source,c('male','other_male','fem1','fem2',
#         'all','fem1+fem2','joint'))) %>%
#     group_by(sid,source,test_correct,train_condition) %>%
#     summarize(value=mean(value))

# ggplot($df,aes(x=source,y=value,color=test_correct)) +
#     stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
#         position=position_nudge(x=0.3)) +
#     geom_point(alpha=0.5,position=position_jitter(width=0.1)) +
#     geom_abline(intercept=0,slope=0,linetype=2) +
#     scale_color_brewer(palette='Set1') +
#     facet_grid(train_condition~.,labeller=label_context)

# ggsave(file.path($dir,"during_target_delta_decode.pdf"),width=11,height=8)

# """
