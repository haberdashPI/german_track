#=
- start decoding from switch
- start from target
- what to do with male_other?
- behavioral true/false is a high bar
  let's try to just figure out how things go from, e.g. the first switch
  also worth looking at "all vs. 1" decoding instead of 1+1 vs. 1 decoding
  just because a speaker isn't attended doesn't mean they can't respond
  and just because it is attended doesn't mean they will respond

what about other ways of looking at the behavioral responses

- test left/right and pseakers with opposite conditions (and global)

# TODO: merge below comments with above comments

# things to count up:
# - first switch, before target
# - does the attended speaker depend on the switches?
# - does the "buildup-up curve" help us predict the behavioral data?

# Do the same analysis for:
# 1. mixture envelope (with and without the target speaker?? may not matter)
# 2. the audiospect envelope
# 3. left vs. right on the feature condition
# 4. both LvR and Speakers for the global condition

=#

include(joinpath(@__DIR__,"..","util","setup.jl"))
# using Gadfly, Cairo, Fontconfig
using DependentBootstrap
# using Makie
using Unitful
using DataKnots
using Tables
using Dates

plot_dir = joinpath(@__DIR__,"..","plots","results_$(Date(now()))")
isdir(plot_dir) || mkdir(plot_dir)

# - train on correct trials only
# - train at target switches

stim_info = JSON.parsefile(joinpath(stimulus_dir,"config.json"))
eeg_files = filter(x -> occursin(r"_mcca65\.bson$",x),readdir(data_dir))
sidfile(id) = @sprintf("eeg_response_%03d_mcca65.bson",id)

############################################################
# speaker analysis

########################################
# anlaysis

# TODO: we don't need this file format, we can use the 65 components directly,
# to reduce memory load.
method = OnlineMethod(window=250ms,lag=250ms,estimation_length=10s,γ=2e-3)
speakers = SpeakerStimMethod(envelope_method=:rms)

data = train_stimuli(method,speakers,eeg_files,stim_info,
    train = "none" => no_indices,
    test = "all_object" => row -> row.condition == "object" ?
        all_indices : no_indices,
    skip_bad_trials = true)

@save joinpath(data_dir,"test_online_speakers.bson") data
# @load joinpath(data_dir,"test_online_rms.bson") data
data = DataFrame(convert(Array{OnlineResult},data))

# TODO: include the 'other' speaker
# to see how well it compare

########################################
# individual plot

main = Scene();
sid8 = @query(data, filter((sid == 8) & (trial <= 75))) |> DataFrame
trials = []
for trial in groupby(sid8,:trial)
   push!(trials,plottrial(method,eachrow(trial),stim_info,sidfile(data.sid[1])))
end
Makie.save(joinpath(plot_dir,"online_test_sid08_1.png"),
    vbox(map(x -> hbox(x...), Iterators.partition(trials,6))...));

sid8 = @query(data, filter((sid == 8) & (trial > 75))) |> DataFrame
trials = []
for trial in groupby(sid8,:trial)
   push!(trials,plottrial(method,eachrow(trial),stim_info,sidfile(data.sid[1])))
end
Makie.save(joinpath(plot_dir,"online_test_sid08_2.png"),
    vbox(map(x -> hbox(x...), Iterators.partition(trials,6))...));

main = Scene();
sid9 = @query(data, filter((sid == 9) & (trial <= 75))) |> DataFrame
trials = []
for trial in groupby(sid9,:trial)
   push!(trials,plottrial(method,eachrow(trial),stim_info,sidfile(data.sid[1])))
end
Makie.save(joinpath(plot_dir,"online_test_sid09_1.png"),
    vbox(map(x -> hbox(x...), Iterators.partition(trials,6))...));

sid9 = @query(data, filter((sid == 9) & (trial > 75))) |> DataFrame
trials = []
for trial in groupby(sid9,:trial)
   push!(trials,plottrial(method,eachrow(trial),stim_info,sidfile(data.sid[1])))
end
Makie.save(joinpath(plot_dir,"online_test_sid09_2.png"),
    vbox(map(x -> hbox(x...), Iterators.partition(trials,6))...));


# stim_events, = events_for_eeg(sidfile(row.sid),stim_info)

########################################
# summary plot

dfat = by(data,:sid) do dfsid
    stim_events, = events_for_eeg(sidfile(dfsid.sid[1]),stim_info)
    by(dfsid,:trial) do dftrial
        attend = speakerattend(dftrial,stim_events,stim_info,
            ustrip(uconvert(Hz,1/method.params.window)))
        DataFrame(
            targetattend = attend,
            test_correct = dftrial.test_correct[1],
            condition = dftrial.condition[1]
        )
    end
end

dfat_mean = by(dfat,[:test_correct,:sid,:condition],
    :targetattend => function(x)
        lower,upper = dbootconf(copy(x),bootmethod=:iid,alpha=0.25)
        (mean=mean(x),lower=lower,upper=upper)
    end)

plot(dfat_mean,x=:test_correct,y=:mean,ymin=:lower,ymax=:upper,
    xgroup=:sid,Geom.subplot_grid(Geom.errorbar,Geom.point)) |>
    PDF(joinpath(plot_dir,"attend_speakers.pdf"),8inch,4inch)

############################################################
# channel analysis
online = OnlineMethod(window=250ms,lag=250ms,estimation_length=10s,γ=2e-3)
channels = ChannelStimMethod(envelope_method=:rms)

data = train_stimuli(online,channels,eeg_files,stim_info,
    train = "none" => no_indices,
    test = "all_feature" => row -> row.condition == "feature" ?
        all_indices : no_indices,
    skip_bad_trials = true)

@save joinpath(data_dir,"test_online_channels.bson") data
# @load joinpath(data_dir,"test_online_channels.bson") data
data = DataFrame(convert(Array{OnlineResult},data))

########################################
# summary plot

dfat = by(data,:sid) do dfsid
    stim_events, = events_for_eeg(sidfile(dfsid.sid[1]),stim_info)
    by(dfsid,:trial) do dftrial
        attend = channelattend(dftrial,stim_events,stim_info,
            ustrip(uconvert(Hz,1/online.params.window)))
        DataFrame(
            targetattend = attend,
            test_correct = dftrial.test_correct[1],
            condition = dftrial.condition[1]
        )
    end
end

dfat_mean = by(dfat,[:test_correct,:sid,:condition],
    :targetattend => function(x)
        lower,upper = dbootconf(copy(x),bootmethod=:iid,alpha=0.25)
        (mean=mean(x),lower=lower,upper=upper)
    end)

plot(dfat_mean,x=:test_correct,y=:mean,ymin=:lower,ymax=:upper,
    xgroup=:sid,Geom.subplot_grid(Geom.errorbar,Geom.point)) |>
    PDF(joinpath(plot_dir,"attend_channels.pdf"),8inch,4inch)

############################################################
# first switch speaker analysis

########################################
# anlaysis

# TODO: we don't need this file format, we can use the 65 components directly,
# to reduce memory load.
method = OnlineMethod(window=250ms,lag=250ms,estimation_length=10s,γ=2e-3)
speakers = SpeakerStimMethod(envelope_method=:rms)

switch_times =
    convert(Array{Array{Float64}},stim_info["test_block_cfg"]["switch_times"])
fs = convert(Float64,stim_info["fs"])
first_switch = map(x -> (x[1]/fs,10.0),switch_times)

data = train_stimuli(method,speakers,eeg_files,stim_info,
    train = "none" => no_indices,
    test = "first_switch" => row -> row.condition == "object" ?
        first_switch[row.sound_index] : no_indices,
    skip_bad_trials = true)

@save joinpath(data_dir,"test_online_first_switch_speakers.bson") data
# @load joinpath(data_dir,"test_online_rms.bson") data
data = DataFrame(convert(Array{OnlineResult},data))



alert()
