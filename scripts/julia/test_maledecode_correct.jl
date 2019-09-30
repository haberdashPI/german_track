# question to adress: can we use the male decoder to predict correct responses
# for male targets? We would expect this to be equivalent during the global
# (test) condition for the framel decoder for female targets

using DrWatson; quickactivate(@__DIR__,"german_track")
using GermanTrack

stim_info = JSON.parsefile(joinpath(stimulus_dir(),"config.json"))
eeg_files = filter(x -> occursin(r"_mcca34\.mcca_proj$",x),readdir(data_dir()))
# eeg_files = eeg_files[1:1]

encoding = JointEncoding(PitchSurpriseEncoding(),ASEnvelope())

target_times =
    convert(Array{Float64},stim_info["test_block_cfg"]["target_times"])

before_target = map(target_times) do time
    iszero(time) ? no_indices : only_near(time,10,window=(-1.5,0))
end

speakers = convert(Array{Int},
    stim_info["test_block_cfg"]["trial_target_speakers"])

correct_male_target = "correct_male_target" =>
    @λ(_row.condition == "test" && _row.correct &&
        speakers[_row.sound_index] == 1 ?
        before_target[_row.sound_index] : no_indices)

correct_fem_target = "correct_fem_target" =>
    @λ(_row.condition == "test" && _row.correct &&
        speakers[_row.sound_index] == 2 ?
        before_target[_row.sound_index] : no_indices)

male_target = "male_target" =>
    @λ(_row.condition == "test" && speakers[_row.sound_index] == 1 ?
        before_target[_row.sound_index] : no_indices)

fem_target = "fem_target" =>
    @λ(_row.condition == "test" && speakers[_row.sound_index] == 2 ?
        before_target[_row.sound_index] : no_indices)


df, encodings, decoders = train_test(
    StaticMethod(NormL2(0.2),cor),
    SpeakerStimMethod(
        encoding=encoding,
        sources=["male","fem1","fem2","male-fem1-fem2","male_other"]),
    resample = 64,
    eeg_files, stim_info,
    return_encodings = true,
    train = [
        correct_male_target; correct_male_target;
        correct_fem_target; correct_fem_target
    ],
    test = [
        male_target; fem_target;
        male_target; fem_target
    ]
)
alert()

function addconds!(df)
    if :condition_str ∉ names(df)
        df[!,:condition_str] = df.condition
    end
    df[!,:train] = replace.(df.condition_str,
        Ref(r"train-(correct_[a-z]+)_.*" => s"before_\1"))
    df[!,:test] = replace.(df.condition_str,
        Ref(r".*test-([a-z]+)_.*" => s"before_\1"))
    df
end

df = addconds!(df)
encodings = addconds!(encodings)
decoders = addconds!(decoders)

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

R"""

library(tidyr)
library(dplyr)
library(ggplot2)

df = $df %>% filter(source %in% c('male','fem1','fem2')) %>%
    select(-condition_str) %>%
    rename(decoded_source = source)

ggplot(df, aes(x=test,y=value,color=test_correct)) +
    stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
        position=position_nudge(x=0.3)) +
    geom_point(alpha=0.5,position=position_jitter(width=0.1)) +
    scale_color_brewer(palette='Set1') +
    facet_grid(train+decoded_source~sid,
        labeller=label_both)

ggsave(file.path($dir,"train_before_gendered_target_window.pdf"))

dfmatch_ish = df %>% filter((train == 'before_correct_male' &
                    test == 'before_male') |
                   (train == 'before_correct_fem' &
                    test == 'before_fem'))

ggplot(dfmatch_ish, aes(x=decoded_source,y=value,color=test_correct)) +
    stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
        position=position_nudge(x=0.3)) +
    geom_point(alpha=0.5,position=position_jitter(width=0.1)) +
    scale_color_brewer(palette='Set1') +
    facet_grid(sid~test)

ggsave(file.path($dir,"compare_decode_source_with_matched_train_test.pdf"),
    width=6,height=15)

dfmatchish_sum = dfmatch_ish %>%
    group_by(test,decoded_source,test_correct,sid) %>%
    summarize(value=mean(value))

ggplot(dfmatchish_sum, aes(x=decoded_source,y=value,color=test_correct)) +
    stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
        position=position_nudge(x=0.3)) +
    geom_point(alpha=0.5,position=position_jitter(width=0.1)) +
    scale_color_brewer(palette='Set1') +
    facet_grid(~test)

ggsave(file.path($dir,"mean_compare_decode_source_with_matched_train_test.pdf"),
    width=6,height=4)

dfmatch = df %>% filter((decoded_source == 'male' &
                    train == 'before_correct_male' &
                    test == 'before_male') |
                   (decoded_source == 'fem1' &
                    train == 'before_correct_fem' &
                    test == 'before_fem'))

ggplot(dfmatch, aes(x=test,y=value,color=test_correct)) +
    stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
        position=position_nudge(x=0.3)) +
    geom_point(alpha=0.5,position=position_jitter(width=0.1)) +
    scale_color_brewer(palette='Set1') +
    facet_grid(~sid)

ggsave(file.path($dir,"train_before_gendered_target_window_matched.pdf"),
    width=11,height=5)

"""

enc12 = map(eachrow(encodings[encodings.sid .== 12,:])) do row
    vcat(
        DataFrame(stim = row.stim[:,1], pred = row.pred[:,1],
            source = row.source,
            correct = row.test_correct,
            time = axes(row.stim,1) ./ 64,
            trial = row.trial,
            train = row.train, test = row.test, feature = "pitch"),
        DataFrame(stim = row.stim[:,2], pred = row.pred[:,2],
            source = row.source,
            trial = row.trial,
            correct = row.test_correct,
            time = axes(row.stim,1) ./ 64,
            train = row.train, test = row.test, feature = "envelope")
    )
end |> @λ(reduce(vcat,_))

R"""

dfenc = $enc12 %>%
    filter((source == 'male' &
            train == 'before_correct_male' &
            test == 'before_male') |
            (source == 'fem1' &
            train == 'before_correct_fem' &
            test == 'before_fem')) %>%
    group_by(source,trial,time,train,test,feature,correct) %>%
    gather(stim,pred,key="kind",value="level")

dfenc = dfenc %>%
    group_by(kind,feature) %>%
    mutate(slevel = level / mad(level))

male_trial_ranges = split(sort(unique(filter(dfenc,source == 'male')$trial)),rep(1:4,each=10))
fem_trial_ranges = split(sort(unique(filter(dfenc,source == 'fem1')$trial)),rep(1:4,each=10))
i = 1

ggplot(filter(dfenc,trial %in% c(male_trial_ranges[[i]],fem_trial_ranges[[i]])),
    aes(x=time,y=slevel,color=kind)) +
    geom_line() +
    facet_grid(feature~source+trial,scales='free')

ggsave(file.path($dir,"envelope_encodings.pdf"),
    width=11,height=5)

# TODO: show some stats of these encodings

dfenc = $enc12 %>%
    group_by(source,trial,time,train,test,feature,correct) %>%
    gather(stim,pred,key="kind",value="level")

ggplot(filter(dfenc,kind=='stim'),aes(x=source,y=level)) +
    geom_violin(width=0.5,position=position_nudge(x=0.5)) +
    geom_point(alpha=0.02,position=position_jitter(width=0.1)) +
    facet_wrap(~feature,scales='free_y') +
    scale_color_brewer(palette='Set1')
ggsave(file.path($dir,"feature_encoding_distribution.pdf"),width=8,height=4)

ggplot(filter(dfenc,kind=='stim'),aes(x=source,y=level)) +
    geom_violin(width=0.5,position=position_nudge(x=0.5)) +
    geom_point(alpha=0.02,position=position_jitter(width=0.1)) +
    facet_grid(feature+correct~train+test,scales='free_y') +
    scale_color_brewer(palette='Set1')

ggsave(file.path($dir,"feature_encoding_distribution_bycategory.pdf"),width=8,height=4)

feature_summaries = dfenc %>%
    group_by(kind,source,train,test,feature,correct) %>%
    summarize(scale = sd(level), rscale = mad(level))

ggplot(feature_summaries,aes(x=source,y=rscale)) +
    geom_point(alpha=0.5,position=position_jitter(width=0.2)) +
    stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
        position=position_nudge(x=0.3)) +
    facet_grid(feature~kind,scales='free_all') +
    scale_color_brewer(palette='Set1')

ggplot(filter(feature_summaries,kind=='stim'),aes(x=source,y=rscale)) +
    geom_point(alpha=0.5,position=position_jitter(width=0.2)) +
    stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
        position=position_nudge(x=0.3)) +
    facet_wrap(~feature,scales='free_y') +
    scale_color_brewer(palette='Set1')

ggsave(file.path($dir,"feature_encoding_variance.pdf"),width=8,height=4)
"""



