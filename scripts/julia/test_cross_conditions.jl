
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

const speakers = convert(Array{Int},
    stim_info["test_block_cfg"]["trial_target_speakers"])
const tindex = Dict("male" => 1, "fem" => 2)
const cond_label = Dict("object" => "object", "global" => "test")

listen_conds = ["object","global"]
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

df, encodings, decoders = train_test(
    StaticMethod(NormL2(0.2),cor),
    SpeakerStimMethod(
        encoding=encoding,
        sources=["male-fem1-fem2","male-fem1-fem2_other"]),
    resample = 64,
    eeg_files, stim_info,
    return_encodings = true,
    train = repeat(outer=2,[
        join(("correct",cond,target),"_") =>
            conditions[join(("correct",cond,target),"_")]
        for cond in listen_conds, target in targets
    ]),
    test = [
        [
            join(("all",cond,target),"_") =>
                conditions[join(("all",cond,target),"_")]
            for cond in listen_conds, target in targets
        ];
        [
            join(("all",cond,target),"_") =>
                conditions[join(("all",cond,target),"_")]
            for cond in reverse(listen_conds), target in targets
        ];
    ]
)
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

df = $df %>%
    select(-condition_str)

ggplot(df,aes(x=test_target,y=value,color=test_correct)) +
    stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
        position=position_nudge(x=0.3)) +
    geom_point(alpha=0.5,position=position_jitter(width=0.1)) +
    scale_color_brewer(palette='Set1') +
    facet_grid(train_condition+test_condition~sid+source,labeller=label_context)

ggsave(file.path($dir,"test_across_conditions.pdf"))
"""

