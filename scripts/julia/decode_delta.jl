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

# TODO: using the simpler approach of collecting on the data transparently
# should get rid of a lot of old code; and serve as a double check on any bugs

# the plan is to first look at the indices that are actually
# being trained and tested vs. the folds
N = sum(@λ(size(_subj.events,1)), values(subjects))
progress = Progress(length(sources)*N,desc="Assembling Data: ")
df = mapreduce(vcat,values(subjects)) do subject
    rows = filter(1:size(subject.events,1)) do i
        !subject.events.bad_trial[i]
    end

    mapreduce(vcat,rows) do row
        si = subject.events.sound_index[row]
        event = subject.events[row,[:correct,:target_present,:target_source,
            :condition,:trial,:sound_index,:target_time]] |> copy

        window = only_near(event.target_time,fs,window=(0,0.5))

        mapreduce(vcat,sources) do source
            stim, = load_stimulus(source,event,stim_encoding,fs,stim_info)
            maxlen = min(size(subject.eeg[row],2),size(stim,1))
            ixs = bound_indices(window,fs,maxlen)

            next!(progress)
            DataFrame(;
                event...,
                eeg = [view(subject.eeg[row]',ixs,:)],
                stim = [view(stim,ixs,:)],
                source = string(source),
                sid = subject.sid
            )
        end
    end
end

# santity check, train all data in each condition, and then test all
# data (then implement k-fold validation)
df.index = 1:size(df,1)
dftarget = df[(df.target_present .== true),:]
K = 10
N = groupby(dftarget, [:condition,:source,:target_source]) |> length
progress = Progress(K*N,desc="Building decoders: ")
models = by(dftarget, [:condition,:source,:target_source]) do sdf
    test_indices = sdf.index
    train_indices = sdf.index[(sdf.correct .== true),:]
    mapreduce(vcat,folds(K,train_indices,test_indices)) do (trainfold,testfold)
        ix = findall(in.(sdf.index,Ref(trainfold)))
        model = decoder(NormL2(0.2),reduce(vcat,sdf.stim[ix]),
            reduce(vcat,sdf.eeg[ix]), 0:nlags)
        next!(progress)
        DataFrame(model = [model],indices = [testfold])
    end
end

only(x) = length(x) == 1 ? first(x) : error("expected single element")

# TODO: now use k-fold cross validation

# TODO: fix an issue with the trial labels for SID 14
cols = [:correct,:sid,:condition,:source,:target_source]
N = sum(length,models.indices)
results = mapreduce(vcat,eachrow(models)) do modeled
    mapreduce(vcat,modeled.indices) do index
        pred = decode(df.eeg[index], modeled.model, 0:nlags)
        C = only(cor(vec(pred),vec(df.stim[index])))
        DataFrame(C = C;copy(df[index,cols])...)
    end
end
source_names = ["male", "female"]
results.target_source = get.(Ref(source_names),Int.(results.target_source),missing)


dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

R"""

library(tidyr)
library(dplyr)
library(ggplot2)
library(stringr)

plotdf = $(results) %>%
    mutate(source = factor(source,c('male','other_male','fem1','fem2',
        'all','fem1+fem2','joint'))) %>%
    group_by(sid,source,correct,condition) %>%
    summarize(C=mean(C))

ggplot(plotdf,aes(x=source,y=C,color=correct)) +
    stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
        position=position_nudge(x=0.3)) +
    geom_point(alpha=0.5,position=position_jitter(width=0.1)) +
    geom_abline(intercept=0,slope=0,linetype=2) +
    scale_color_brewer(palette='Set1') +
    facet_grid(condition~.,labeller=label_context)

ggsave(file.path($dir,"during_target_delta_decode.pdf"),width=11,height=8)

"""
