using DrWatson; quickactivate(@__DIR__,"german_track")
using GermanTrack
using AxisArrays
using PlotAxes

stim_info = JSON.parsefile(joinpath(stimulus_dir(),"config.json"))
eeg_files = filter(x -> occursin(r"_mcca34\.mcca_proj$",x),readdir(data_dir()))
eeg_files = eeg_files[1:1]

encoding = JointEncoding(PitchSurpriseEncoding(),ASEnvelope())

# gathering stimlus information
target_times =
    convert(Array{Float64},stim_info["test_block_cfg"]["target_times"])

during_target = map(target_times) do time
    iszero(time) ? no_indices : only_near(time,10,window=(0,1.5))
end
non_target = map(target_times) do time
    iszero(time) ? no_indices : not_near(time,10,window=(-0.5,1.5))
end

speakers = convert(Array{Int},
    stim_info["test_block_cfg"]["trial_target_speakers"])


# THOUGHT: can we use z-scored correlation as a way of determining
# an overall, detectability score?

# defining conditions

test_target = "test_target" =>
    @λ(_row.condition == "test" ?
        during_target[_row.sound_index] : no_indices)

test_nontarget = "test_nontarget" =>
    @λ(_row.condition == "object" ?
        non_target[_row.sound_index] : no_indices)

male_test_target = "male_test_target" =>
    @λ(_row.condition == "test" && speakers[_row.sound_index] == 1 ?
        during_target[_row.sound_index] : no_indices)

male_test_nontarget = "male_test_nontarget" =>
    @λ(_row.condition == "test" && speakers[_row.sound_index] == 1 ?
        non_target[_row.sound_index] : no_indices)

female_test_target = "female_test_target" =>
    @λ(_row.condition == "test" && speakers[_row.sound_index] == 2 ?
        during_target[_row.sound_index] : no_indices)

female_test_nontarget = "female_test_nontarget" =>
    @λ(_row.condition == "test" && speakers[_row.sound_index] == 2 ?
        non_target[_row.sound_index] : no_indices)

object_target = "object_target" =>
    @λ(_row.condition == "object" ?
        during_target[_row.sound_index] : no_indices)

object_nontarget = "object_nontarget" =>
    @λ(_row.condition == "object" ?
        non_target[_row.sound_index] : no_indices)

male_object_target = "male_object_target" =>
    @λ(_row.condition == "object" && speakers[_row.sound_index] == 1 ?
        during_target[_row.sound_index] : no_indices)

male_object_nontarget = "male_object_nontarget" =>
    @λ(_row.condition == "object" && speakers[_row.sound_index] == 1 ?
        non_target[_row.sound_index] : no_indices)

female_object_target = "female_object_target" =>
    @λ(_row.condition == "object" && speakers[_row.sound_index] == 2 ?
        during_target[_row.sound_index] : no_indices)

female_object_nontarget = "female_object_nontarget" =>
    @λ(_row.condition == "object" && speakers[_row.sound_index] == 2 ?
        non_target[_row.sound_index] : no_indices)


df, decoders = train_test(
    StaticMethod(NormL2(0.2),cor),
    SpeakerStimMethod(encoding=encoding),
    resample = 64,
    eeg_files, stim_info,

    train = [
        male_test_target;
        male_test_target;
        female_test_target;
        female_test_target;

        male_object_target;
        male_object_target;
        female_object_target;
        female_object_target;
    ],

    test = [
        male_test_target;
        male_test_nontarget;
        female_test_target;
        female_test_nontarget;

        male_object_target;
        male_object_nontarget;
        female_object_target;
        female_object_nontarget;
    ]
)
alert()

pattern = r"^.*test-([a-z]+)_([a-z]+)_([a-z]+).*$"
function addconds!(df)
    if :condition_str ∉ names(df)
        df[!,:condition_str] = df.condition
    end
    df[!,:test] = replace.(df.condition_str,
        Ref(pattern => s"\3"))
    df[!,:target_source] = replace.(df.condition_str,
        Ref(pattern => s"\1"))
    df.condition = replace.(df.condition_str,
        Ref(pattern => s"\2"))
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

df = $df %>%
    select(-condition_str) %>%
    filter(source != "all-male") %>%
    group_by(source,index,sid,condition,target,test_correct) %>%
    spread(test,value)

ggplot(df,
    aes(x=nontarget,y=target,color=target_source)) +
    geom_point(alpha=0.5) +
    geom_abline(slope=1,intercept=0,linetype=2) +
    scale_color_brewer(palette='Set1') +
    facet_grid(condition~source)

ggsave(file.path($dir,"fem_v_male_near_target.pdf"),width=11,height=6)

ggplot(df,
    aes(x=condition,y=target-nontarget,color=target_source)) +
    geom_point(alpha=0.5,position=
        position_jitterdodge(dodge.width=0.2,jitter.width=0.1)) +
    stat_summary(fun.data='mean_cl_boot',#fun.args=list(conf.int=0.75),
        position=position_nudge(x=0.3)) +
    scale_color_brewer(palette='Set1') +
    facet_grid(~source)

ggsave(file.path($dir,"fem_v_male_near_target_diff.pdf"),width=11,height=6)

"""

# TODO: plot the coefficients (just show the matrix of values for now)
# worry about interpreting as points on the scalp later on

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

ggplot(filter($all_coefs,source != 'male-fem1-fem2'),
    aes(y=component,x=lag,fill=value)) + geom_raster() +
    facet_grid(test+feature~condition+source) +
    scale_fill_distiller(palette='RdBu')

ggsave(file.path($dir,"coefs_fem_v_male_near_target.pdf"),width=11,height=8)

ggplot(filter($all_coefs,source == 'male-fem1-fem2'),
    aes(y=component,x=lag,fill=value)) + geom_raster() +
    facet_grid(feature~condition+test) +
    scale_fill_distiller(palette='RdBu')

ggsave(file.path($dir,"coefs_fem_v_male_near_target_join_sources.pdf"),width=11,height=8)

"""
