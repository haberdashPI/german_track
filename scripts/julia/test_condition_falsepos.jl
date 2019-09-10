
# premise: can we find the false positives on the basis
# of a decoder trained on the correctly identified targets

# not quite...

# how do we handle that there were looking for any differences across
# conditions?? the problem is there will naturally be fewer cases to test in
# the object condition, factoring out the differences

# so: 1.) we should be able to predict the actual correct/incorrect
# responses for the catch trials in both conditions
# if so, there is something different in the response
# this means we should look at the coefficients in this case

using DrWatson; quickactivate(@__DIR__,"german_track")
using GermanTrack
using AxisArrays
using PlotAxes

stim_info = JSON.parsefile(joinpath(stimulus_dir(),"config.json"))
eeg_files = filter(x -> occursin(r"_mcca34\.mcca_proj$",x),readdir(data_dir()))
eeg_files = eeg_files[1:1]

encoding = JointEncoding(PitchSurpriseEncoding(),ASEnvelope())

test_truepos = "test_truepos" =>
    @λ(_row.condition == "test" && _row.correct && _row.target_present ?
        all_indices : no_indices)

object_truepos = "object_truepos" =>
    @λ(_row.condition == "object" && _row.correct && _row.target_present ?
        all_indices : no_indices)

test_target_trials = "test_target_trials" =>
    @λ(_row.condition == "test" && _row.target_present ?
        all_indices : no_indices)

object_catch_trials = "object_catch_trials" =>
    @λ(_row.condition == "object" && _row.target_present ?
        all_indices : no_indices)


test_catch_trials = "test_catch_trials" =>
    @λ(_row.condition == "test" && !_row.target_present ?
        all_indices : no_indices)

object_target_trials = "object_target_trials" =>
    @λ(_row.condition == "object" && _row.target_present ?
        all_indices : no_indices)

object_catch_trials = "object_catch_trials" =>
    @λ(_row.condition == "object" && !_row.target_present ?
        all_indices : no_indices)


df, decoders = train_test(
    StaticMethod(NormL2(0.2),cor),
    SpeakerStimMethod(encoding=encoding),
    resample = 64,
    eeg_files, stim_info,

    train = [
        test_truepos;test_truepos;
        object_truepos;object_truepos
    ],

    test = [
        test_target_trials;test_catch_trials;
        object_target_trials;object_catch_trials;
    ]
)
alert()

pattern = r"^.*test-([a-z]+)_([a-z_]+).*$"
function addconds!(df)
    if :condition_str ∉ names(df)
        df[!,:condition_str] = df.condition
    end
    df[!,:test] = replace.(df.condition_str,
        Ref(pattern => s"\2"))
    df.condition = replace.(df.condition_str,
        Ref(pattern => s"\1"))
    df
end

df = addconds!(df)
decoders = addconds!(decoders)

speakers = convert(Array{Int},
    stim_info["test_block_cfg"]["trial_target_speakers"])
df[!,:target_source] = map(speakers[df.stim_id]) do index
    if index > 0
        ["male","fem"][index]
    else
        "none"
    end
end

df[!,:response] = ifelse.((df.target_source .== "none") .== df.test_correct,
    "nontarget","target")
categorical!(df,:response)

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

R"""

library(tidyr)
library(dplyr)
library(ggplot2)

df = $df %>% select(-condition_str)

ggplot(df,
    aes(y=value,x=test,color=interaction(response,target_source))) +
    geom_point(alpha=0.3,position=
        position_jitterdodge(dodge.width=0.2,jitter.width=0.1)) +
    stat_summary(fun.data='mean_cl_boot',fun.args=list(conf.int=0.75),
        position=position_nudge(x=0.3)) +
    facet_grid(source~condition) +
    scale_color_brewer(palette='Paired',name='resp, target source')

ggsave(file.path($dir,"catch_trial_prediction.pdf"),width=6,height=9)

"""

# TODO: generalize this code and make it par tof `train_test`
quant = (100,100,10)
# quant = (10,10,10)
all_coefs = mapreduce(vcat,eachrow(decoders)) do row
    if size(row.coefs,3) == 2
        coefs, = PlotAxes.asplotable(AxisArray(
            row.coefs,
            Axis{:component}(Base.axes(row.coefs,1)),
            Axis{:lag}(Base.axes(row.coefs,2)),
            Axis{:feature}([:envelop,:pitch])
        ), quantize = quant)
    elseif size(row.coefs,3) == 1
        coefs, = PlotAxes.asplotable(AxisArray(
            row.coefs,
            Axis{:component}(Base.axes(row.coefs,1)),
            Axis{:lag}(Base.axes(row.coefs,2)),
            Axis{:feature}([:envelop])
        ), quantize = quant)
    else
        coefs, = PlotAxes.asplotable(AxisArray(
            row.coefs,
            Axis{:component}(Base.axes(row.coefs,1)),
            Axis{:lag}(Base.axes(row.coefs,2)),
            Axis{:feature}([
                Symbol(string(feature,source))
                for feature = [:envelop,:pitch],
                    source = [:male,:fem1,:fem2]
            ])
        ), quantize = quant)
    end
    for col in setdiff(names(row),[:coefs])
        coefs[!,col] .= row[col]
    end
    coefs
end

R"""

library(tidyr)
library(dplyr)
library(ggplot2)

ggplot(filter($all_coefs,source == 'male-fem1-fem2'),
    aes(y=component,x=lag,fill=value)) + geom_raster() +
    facet_grid(feature~condition+source) +
    scale_fill_distiller(palette='RdBu')

ggsave(file.path($dir,"catch_trial_prediction_features.pdf"),width=5,height=9)

"""

ggsave(file.path($dir,"coefs_fem_v_male_near_target.pdf"),width=11,height=8)
