# question: can we use the male decoder to predict correct responses for male
# targets: we would expect this to be equivalent during the global (test)
# condition for the framel decoder for female targets


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
    SpeakerStimMethod(encoding=encoding),
    resample = 64,
    eeg_files, stim_info,
    return_encodings = true,
    train = [ correct_male_target; correct_fem_target ],
    test = [ male_target; fem_target ]
)
alert()

function addconds!(df)
    if :condition_str ∉ names(df)
        df[!,:condition_str] = df.condition
    end
    df[!,:target_source] = replace.(df.condition_str,
        Ref(r".*test-([a-z]+)_.*" => s"\1"))
    df
end

df = addconds!(df)
decoders = addconds!(decoders)

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

R"""

library(tidyr)
library(dplyr)
library(ggplot2)

df = $df %>% filter(source %in% c('male','female')) %>%
    select(-condition_str)

ggplot(df, aes(x=target_source,y=value,color=test_correct)) +
    stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
        position=position_nudge(x=0.3)) +
    geom_point(alpha=0.5,position=position_jitter(width=0.1)) +
    scale_color_brewer(palette='Set1') +
    facet_grid(source~sid)

ggsave(file.path($dir,"source_target_decode_test_correct.pdf"))

df = df %>% filter((source == 'male' && target_source == 'male') ||
                   (source == 'female' & target_source == 'fem'))

ggplot(df, aes(x=target_source,y=value,color=test_correct)) +
    stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
        position=position_nudge(x=0.3)) +
    geom_point(alpha=0.5,position=position_jitter(width=0.1)) +
    scale_color_brewer(palette='Set1') +
    facet_grid(~sid)

ggsave(file.path($dir,"source_target_decode_test_correct__compare_sources.pdf"))

"""

# TODO: plot envelopes, pitches
