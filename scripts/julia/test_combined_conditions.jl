using DrWatson; quickactivate(@__DIR__,"german_track"); using GermanTrack

stim_info = JSON.parsefile(joinpath(stimulus_dir(),"config.json"))
eeg_files = filter(x -> occursin(r"_mcca34\.mcca_proj$",x),readdir(data_dir()))
# eeg_files = eeg_files[1:1]

encoding = JointEncoding(PitchSurpriseEncoding(), ASEnvelope())
# eeg_encoding = JointEncoding(RawEncoding(),
#     FilteredPower("alpha",5,15),
#     FilteredPower("gamma",30,100)
# )

target_times =
    convert(Array{Float64},stim_info["test_block_cfg"]["target_times"])

before_target = map(target_times) do time
    iszero(time) ? no_indices : only_near(time,10,window=(-1.5,0.5))
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

df = train_test(
    StaticMethod(NormL2(0.2),cor),
    SpeakerStimMethod(
        encoding=encoding,
        sources=[male_source, fem1_source, fem2_source, other(male_source)]),
    # encode_eeg = eeg_encoding,
    resample = 64, # NOTE: resampling occurs after alpha and gamma are encoded
    eeg_files, stim_info,
    maxlag=0.8,
    train = repeat(train_conditions,inner=3),
    test = test_conditions
);
alert()

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

R"""

library(tidyr)
library(dplyr)
library(ggplot2)
library(stringr)

ggplot($df,aes(x=source,y=value,color=test_correct)) +
    stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
        position=position_dodge(width=0.85)) +
    geom_point(alpha=0.5,position=position_jitterdodge(dodge.width=0.3,
        jitter.width=0.1)) +
    scale_color_brewer(palette='Set1') +
    facet_grid(sid~test_condition,labeller=label_context) +
    geom_abline(slope=0,intercept=0,linetype=2)

ggsave(file.path($dir,"cross_conditions.pdf"),width=11,height=8)
"""
