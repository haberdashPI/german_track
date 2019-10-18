using DrWatson; quickactivate(@__DIR__,"german_track"); using GermanTrack

stim_info = JSON.parsefile(joinpath(stimulus_dir(),"config.json"))
eeg_files = filter(x -> occursin(r"_mcca34\.mcca_proj$",x),readdir(data_dir()))
eeg_files = eeg_files[1:1]

encoding = JointEncoding(PitchSurpriseEncoding(), ASEnvelope())
# eeg_encoding = JointEncoding(RawEncoding(),
#     FilteredPower("alpha",5,15),
#     FilteredPower("gamma",30,100)
# )

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

train_conditions = map(targets) do target
    (target=target,) =>
        @λ(_row.correct &&
            speakers[_row.sound_index] == tindex[target] ?
                before_target[_row.sound_index] : no_indices)
end


test_conditions = mapreduce(vcat,listen_conds) do condition
    mapreduce(vcat,targets) do target
        [(condition=condition,target=target) =>
            @λ(_row.condition == cond_label[condition] &&
                speakers[_row.sound_index] == tindex[target] ?
                    before_target[_row.sound_index] : no_indices)]
    end
end

function measures(pred,stim)
    (joint_cor = cor(vec(pred),vec(stim)),
     male_cor = cor(vec(pred[:,1:2]),vec(stim[:,1:2])),
     fem1_cor = cor(vec(pred[:,3:4]),vec(stim[:,3:4])),
     fem2_cor = cor(vec(pred[:,5:6]),vec(stim[:,5:6])))
end

df,models = train_test(
    K = 20,
    StaticMethod(NormL2(0.2),measures),
    SpeakerStimMethod(
        encoding=encoding,
        sources=[joint_source, other(joint_source)]),
    # encode_eeg = eeg_encoding,
    resample = 64, # NOTE: resampling occurs after alpha and gamma are encoded
    eeg_files, stim_info,
    maxlag=0.8,
    return_models = true,
    train = repeat(train_conditions,inner=3),
    test = test_conditions
);
alert()

# TODO: this conditions aren't being properly named
# (fix renaming scheme)
function adjust_columns!(df)
    if :stim_id ∈ names(df)
        df[!,:location] = direction[df.stim_id]
    end
    df
end

df = adjust_columns!(df)
models = adjust_columns!(models)

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

R"""

library(tidyr)
library(dplyr)
library(ggplot2)
library(stringr)

dfmatch = $df %>% filter(source == 'joint') %>%
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

ggsave(file.path($dir,"train_across_conditions.pdf"))

dfmatch_means = dfmatch %>%
    group_by(sid,target_detected,target,condition,featuresof) %>%
    summarize(cor = mean(cor))

ggplot(dfmatch_means,aes(x=featuresof,y=cor,color=target_detected))     +
    stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
        aes(fill=target_detected),pch=21,size=0.5,
        color='black',
        position=position_nudge(x=0.3)) +
    geom_point(alpha=0.5,position=position_jitter(width=0.1)) +
    scale_color_brewer(palette='Set1') +
    scale_fill_brewer(palette='Set1') +
    theme_classic() +
    facet_grid(condition~target,labeller=label_context)

ggsave(file.path($dir,"mean_test_across_conditions.pdf"),width=8,height=6)

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

ggsave(file.path($dir,"test_across_conditions_weighted_spatial.pdf"),
    width=14,height=8)

dfmatch_spatial_means = dfmatch %>%
    group_by(sid,target_detected,target,condition,location,featuresof) %>%
    summarize(cor = mean(cor))

ggplot(dfmatch_spatial_means,aes(x=featuresof,y=cor,
    color=interaction(location,target_detected))) +
    stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
        aes(fill=interaction(location,target_detected)),pch=21,size=0.5,
        color='black',
        position=position_nudge(x=0.3)) +
    geom_point(alpha=0.5,position=position_jitter(width=0.1)) +
    scale_color_brewer(palette='Paired') +
    scale_fill_brewer(palette='Paired') +
    theme_classic() +
    facet_grid(condition~target,labeller=label_context)


ggsave(file.path($dir,"mean_test_across_conditions_weighted_spatial.pdf"))

dfspatial = dfmatch %>%
    filter(featuresof %in% c('male','fem1'), condition == 'spatial') %>%
    mutate(target_source =
        (featuresof == 'fem1' && target == 'fem' && location == 'right') ||
        (featuresof == 'male' && target == 'male' && location == 'right'))

ggplot(dfspatial,aes(x=target_source,y=cor,color=target_detected)) +
    stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
        aes(fill=target_detected),pch=21,size=0.5,
        color='black',
        position=position_nudge(x=0.3)) +
    geom_point(alpha=0.5,position=position_jitter(width=0.1)) +
    scale_color_brewer(palette='Set1') +
    scale_fill_brewer(palette='Set1') +
    theme_classic() +
    facet_grid(~sid)

ggsave(file.path($dir,"spatial_targets.pdf"))

dfspatial_means = dfspatial %>%
    group_by(sid,target_source,target_detected,target,featuresof) %>%
    summarize(cor = mean(cor))

ggplot(dfspatial_means,aes(x=target_source,y=cor,color=target_detected)) +
    stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
        aes(fill=target_detected),pch=21,size=0.5,
        color='black',
        position=position_nudge(x=0.3)) +
    geom_point(alpha=0.5,position=position_jitter(width=0.1)) +
    scale_color_brewer(palette='Set1') +
    scale_fill_brewer(palette='Set1') +
    theme_classic()

ggsave(file.path($dir,"mean_spatial_targets.pdf"),width=4,height=6)

"""
