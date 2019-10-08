
using DrWatson; quickactivate(@__DIR__,"german_track")
using GermanTrack
using AxisArrays
using PlotAxes
using DataFramesMeta
using DependentBootstrap
using Random
using Statistics

stim_info = JSON.parsefile(joinpath(stimulus_dir(),"config.json"))
eeg_files = filter(x -> occursin(r"_mcca34\.mcca_proj$",x),readdir(data_dir()))
# eeg_files = eeg_files[1:1]

encoding = JointEncoding(
    WeightedEncoding([2.0,1.0,1.0],PitchSurpriseEncoding()),
    WeightedEncoding([2.0,1.0,1.0],ASEnvelope())
)

target_times =
    convert(Array{Float64},stim_info["test_block_cfg"]["target_times"])

before_target = map(target_times) do time
    iszero(time) ? no_indices : only_near(time,10,window=(-1.5,0))
end

const speakers = convert(Array{Int},
    stim_info["test_block_cfg"]["trial_target_speakers"])
const tindex = Dict("male" => 1, "fem" => 2)
const cond_label = Dict("object" => "object", "global" => "test", "spatial" => "feature")
const direction = convert(Array{String},
    stim_info["test_block_cfg"]["trial_target_dir"])

listen_conds = ["object","global", "spatial"]
targets = ["male","fem"]
labels = ["correct","all"]

conditions = mapreduce(vcat,listen_conds) do condition
    mapreduce(vcat,targets) do target
        mapreduce(vcat,labels) do label
            [join((label,condition,target),"_") =>
                @λ(_row.condition == cond_label[condition] &&
                    (label == "all" || _row.correct) &&
                    speakers[_row.sound_index] == tindex[target] ?
                        before_target[_row.sound_index] : no_indices)]
        end
    end
end |> Dict

function measures(pred,stim)
    (joint_cor = cor(vec(pred),vec(stim)),
     male_cor = cor(vec(pred[:,1:2]),vec(stim[:,1:2])),
     fem1_cor = cor(vec(pred[:,3:4]),vec(stim[:,3:4])),
     fem2_cor = cor(vec(pred[:,5:6]),vec(stim[:,5:6])))
end

cond_pairs = Iterators.product(listen_conds,listen_conds)
df, encodings, models = train_test(
    StaticMethod(NormL2(0.2),measures),
    SpeakerStimMethod(
        encoding=encoding,
        sources=["male-fem1-fem2","male-fem1-fem2_other"]),
    resample = 64,
    eeg_files, stim_info,
    return_encodings = true,
    maxlag=0.8,
    return_models = true,
    train = [
        join(("correct",cond,target),"_") =>
            conditions[join(("correct",cond,target),"_")]
        for (cond,_) in cond_pairs, target in targets
    ],
    test = [
        join(("all",cond,target),"_") =>
            conditions[join(("all",cond,target),"_")]
        for (_,cond) in cond_pairs, target in targets
]);
alert()

function addconds!(df)
    if :condition_str ∉ names(df)
        df[!,:condition_str] = df.condition
    end
    df[!,:train_condition] = replace.(df.condition_str,
        Ref(r"train-correct_([a-z]+)_.*" => s"\1"))
    df[!,:train_target] = replace.(df.condition_str,
        Ref(r"train-correct_[a-z]+_([a-z]+)_.*" => s"before_correct_\1"))
    df[!,:test_condition] = replace.(df.condition_str,
        Ref(r".*test-all_([a-z]+)_.*" => s"\1"))
    df[!,:test_target] = replace.(df.condition_str,
        Ref(r".*test-all_[a-z]+_([a-z]+)" => s"\1"))
    df[!,:source] = replace.(df.source,
        Ref(r"male-fem1-fem2" => "joint"))
    if :stim_id ∈ names(df)
        df[!,:location] = direction[df.stim_id]
    end
    df
end

df = addconds!(df)
encodings = addconds!(encodings)
models = addconds!(models)

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

R"""

library(tidyr)
library(dplyr)
library(ggplot2)
library(stringr)

df = $df %>%
    select(-condition_str)

ggplot(df,aes(x=test_target,y=joint_cor,color=test_correct)) +
    stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
        position=position_nudge(x=0.3)) +
    geom_point(alpha=0.5,position=position_jitter(width=0.1)) +
    scale_color_brewer(palette='Set1') +
    facet_grid(train_condition+test_condition~sid+source,labeller=label_context)

dfmatch = df %>% filter(source == 'joint', train_condition == test_condition) %>%
    select(-condition) %>%
    rename(condition = test_condition, target = test_target,
        target_detected = test_correct) %>%
    group_by(sid,target_detected,target,condition,stim_id) %>%
    gather(male_cor,fem1_cor,fem2_cor,key='featuresof',value='cor') %>%
    mutate(featuresof = str_replace(featuresof,"(.*)_cor","\\1"))

ggplot(dfmatch,aes(x=featuresof,y=cor,color=target_detected)) +
    stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
        aes(fill=target_detected),pch=21,size=0.5,
        color='black',
        position=position_nudge(x=0.3)) +
    geom_point(alpha=0.5,position=position_jitter(width=0.1)) +
    scale_color_brewer(palette='Set1') +
    scale_fill_brewer(palette='Set1') +
    theme_classic() +
    facet_grid(condition~sid+target,labeller=label_context)

ggsave(file.path($dir,"test_across_conditions_weighted.pdf"))

ggplot(dfmatch,aes(x=featuresof,y=cor,color=interaction(location,target_detected))) +
    stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
        aes(fill=interaction(location,target_detected)),pch=21,size=0.5,
        color='black',
        position=position_nudge(x=0.3)) +
    geom_point(alpha=0.5,position=position_jitter(width=0.1)) +
    scale_color_brewer(palette='Paired') +
    scale_fill_brewer(palette='Paired') +
    theme_classic() +
    facet_grid(condition~sid+target,labeller=label_context)

ggsave(file.path($dir,"test_across_conditions_spatial_weighted.pdf"))
"""

matched = @where(models,
    (:train_condition .== :test_condition) .&
    (:source .== "joint"))

function bootstrap(df,fn,col;conf=0.95,N=10_000)
    # bootstrap across the trials
    med = similar(fn(df[1,col]))
    lower = similar(med)
    upper = similar(med)

    M = size(df,1)
    baserng = MersenneTwister()
    alldata = fn.(df[:,col])
    for i in eachindex(med)
        rng = copy(baserng)
        samples = map(1:N) do _
            mean(alldata[j][i] for j in sample(1:M,M))
        end
        med[i], lower[i], upper[i] =
            quantile(samples,[0.5,conf/2,1-(conf/2)])
    end

    med, lower, upper
end

feature_names = [
    string(feature,"_",source)
    for feature = [:envelop,:pitch],
        source = [:male,:fem1,:fem2]
]

bootlag = by(matched,[:source,:sid,:test_target,:train_condition]) do trials
    result = bootstrap(trials,@λ(mean(_,dims=1)),:coefs)
    function todf(coefs)
        coefs, = PlotAxes.asplotable(coefs,:col,:page)
        rename!(coefs,:col => :lag, :page => :feature)
        coefs[!,:feature] .= feature_names[convert.(Int,coefs.feature)]
        coefs
    end

    med, lower, upper = todf.(result)
    med[!,:lower] = lower.value
    med[!,:upper] = upper.value

    med
end

R"""

ggplot($bootlag,aes(x=lag,y=value,group=train_condition)) +
    facet_grid(feature~sid+test_target) +
    geom_errorbar(aes(ymin=lower,ymax=upper,color=train_condition),size=0.5,width=0.25) +
    geom_point(aes(color=train_condition),size=0.05) +
    geom_line(aes(color=train_condition))

ggsave(file.path($dir,"lags.pdf"),width=11,height=8)
"""

bootcomp = by(matched,[:source,:sid,:test_target,:train_condition]) do trials
    result = bootstrap(trials,@λ(mean(_,dims=2)),:coefs)
    function todf(coefs)
        coefs, = PlotAxes.asplotable(coefs,:row,:page)
        rename!(coefs,:row => :component, :page => :feature)
        coefs[!,:feature] .= feature_names[convert.(Int,coefs.feature)]
        coefs
    end

    med, lower, upper = todf.(result)
    med[!,:lower] = lower.value
    med[!,:upper] = upper.value

    med
end

R"""

ggplot($bootcomp,aes(x=component,y=value,group=train_condition)) +
    facet_grid(feature~sid+test_target) +
    geom_errorbar(aes(ymin=lower,ymax=upper,color=train_condition)) +
    geom_point(aes(color=train_condition),size=0.25) + coord_flip() +
    geom_line(aes(color=train_condition))

ggsave(file.path($dir,"components.pdf"),width=8,height=11)

"""


# TODO: handle bootstrapping
# # TODO: generalize this code and make it part of `train_test`
# # quant = (10,10,10)
# all_coefs = mapreduce(vcat,eachrow(decoders)) do row
#     coefs, = PlotAxes.asplotable(row.coefs, quantize = (100,100,10))
#     rename!(coefs,:row => :component, :col => :lag, :page => :feature)
#     coefs[!,:feature] .= feature_names[coefs.feature]

#     for col in setdiff(names(row),[:coefs])
#         coefs[!,col] .= row[col]
#     end
#     coefs
# end
# all_coefs = addconds!(all_coefs)


# R"""

# df = filter($all_coefs,
#     test_condition == train_condition,
#     (train_target == 'before_correct_male') == (test_target == 'male'))

# for(sid_ in 8:14){

#     ggplot(filter(df,sid == sid_),aes(y=component,x=lag,fill=value)) +
#         geom_raster() +
#         facet_grid(feature~train_condition+test_target) +
#         scale_fill_distiller(palette='RdBu')
#     ggsave(file.path($dir,sprintf("global_v_object_coefs_sid_%02d.pdf",sid_)))

#     dflags = df %>%
#         group_by(feature,train_condition,test_target,sid,lag) %>%
#         summarize(value = mean(value))

#     ggplot(file.path($dir,filter(dflags,sid == sid_),aes(y=value,x=lag,color=train_condition))) +
#         geom_line() +
#         facet_grid(feature~test_target) +
#         scale_fill_distiller(palette='RdBu')
#         ggsave(sprintf("global_v_object_lags_sid_%02d.pdf",sid_))

#     dfcomps = df %>%
#         group_by(feature,train_condition,test_target,sid,component) %>%
#         summarize(value = mean(value))

#     ggplot(filter(dfcomps,sid == sid_),aes(y=value,x=component,color=train_condition)) +
#         geom_line() + coord_flip() +
#         facet_grid(feature~test_target) +
#         scale_fill_distiller(palette='RdBu')
#         ggsave(file.path($dir,sprintf("global_v_object_components_sid_%02d.pdf",sid_)))
# }

# """