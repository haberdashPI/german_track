#=

- next steps:
    - summary of the mean coefficients from start,
        - from switch
        - look at the male voice
        - examine different envelopes for the online analysis
        - all - target
        - is there some advantage of the male speaker
            relative to the other conditions
        - focus on the male speaker in the object condition
    - show all-speaker for the static analysis

summary:
try and fix the male (look at other subjects first)
then, do we see an advantage of the male speaker in the object conditio
at the start? near target (before, within, after???)
# work on these summary statistics NEXT!!

- behavioral true/false is a high bar
  - let's try to just figure out how things go from, e.g. the first switch
  - worth looking at "all vs. 1" decoding instead of 1+1 vs. 1 decoding
  - worth considering a different envelope
  - rethink relationship to behavioral data: just because a speaker isn't
  attended doesn't mean they can't respond and just because it is attended
  doesn't mean they will respond


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

using Pkg; Pkg.activate(joinpath(@__DIR__,".."))
include(joinpath(@__DIR__,"..","util","setup.jl"))

plot_dir = joinpath(@__DIR__,"..","plots","results_$(Date(now()))")
isdir(plot_dir) || mkdir(plot_dir)

if endswith(gethostname(),".cluster")
    addprocs(SlurmManager(length(eeg_files)), partition="CPU", t="02:00:00",
            cpus_per_task=4,enable_threaded_blas=true)
    @everywhere include(joinpath(@__DIR__,"..","util","setup.jl"))
end

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
speakers = SpeakerStimMethod(envelope_method=:audiospect)

data = train_stimuli(method,speakers,eeg_files,stim_info,
    train = "none" => no_indices,
    # progress=progress,
    test = "all_object" => row -> row.condition == "object" ?
        all_indices : no_indices,
    skip_bad_trials = true)

@save joinpath(data_dir,"test_all_online_speakers.bson") data
# @load joinpath(data_dir,"test_all_online_rms.bson") data
data = DataFrame(convert(Array{OnlineResult},data))

# testing...
# TODO: this is technically wrong, since the event file is always for 8
plots = map(unique(data.trial)) do i
    plottrial(method,eachrow(data[data.trial .== i,:]),stim_info,
        sidfile(data.sid[1]),raw=true)
end;

@vlplot() + vcat((hcat(pl...) for pl in Iterators.partition(plots,6))...)


########################################
# individual plot

sid8 = @query(data, filter((sid == 8))) |> DataFrame

# testing...
# TODO: this is technically wrong, since the event file is always for 8
plots = map(unique(sid8.trial)) do i
    plottrial(method,eachrow(sid8[sid8.trial .== i,:]),stim_info,
        sidfile(sid8.sid[1]),raw=true)
end;

# TODO: allow wrapping concatenation of the figures
# or just figure out a good grid to put these in
@vlplot() + vcat((hcat(pl...) for pl in Iterators.partition(plots,3))...)


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
method = OnlineMethod(window=250ms,lag=250ms,estimation_length=10s,
    γ=2e-3,tol=1e-2)
speakers = SpeakerStimMethod(envelope_method=:audiospect)

switch_times =
    convert(Array{Array{Float64}},stim_info["test_block_cfg"]["switch_times"])
fs = convert(Float64,stim_info["fs"])
first_switch = map(enumerate(switch_times)) do (i,times)
    target_time = stim_info["test_block_cfg"]["target_times"][i]
    if target_time > 0
        j = findlast(x -> x/fs < target_time,times)
        if j == nothing
            (0,10.0)
        else
            (times[j]/fs,10.0)
        end
    else
        no_indices
    end
end


# THEN: run different conditions
#   - audiospect envelope
# THEN: run first switch for all participants

data = train_stimuli(method,speakers,eeg_files,stim_info,
    train = "none" => no_indices,
    test = "all_object" => row -> row.condition == "object" ?
        all_indices : no_indices,
    skip_bad_trials = true)

data = DataFrame(convert(Array{OnlineResult},data))
@save joinpath(data_dir,"test_online_first_switch_speakers_audiospect.bson") data
# @load joinpath(data_dir,"test_online_first_switch_speakers_audiospect.bson") data

# TODO: think through this summary (there are some issues also
# noted in the top-level comments. Also worth plotting individual
# data )

sid8 = @query(data, filter((sid == 8))) |> DataFrame

# testing...
# TODO: this is technically wrong, since the event file is always for 8
plots = map(unique(sid8.trial)) do i
    plottrial(method,eachrow(sid8[sid8.trial .== i,:]),stim_info,
        sidfile(sid8.sid[1]),bounds = row -> first_switch[row.sound_index],
        raw=true)
end;

@vlplot() + vcat((hcat(pl...) for pl in Iterators.partition(plots,6))...)

# dfat_mean = by(dfat,[:test_correct,:sid,:condition],
#     :targetattend => function(x)
#         lower,upper = dbootconf(copy(x),bootmethod=:iid,alpha=0.25)
#         (mean=mean(x),lower=lower,upper=upper)
#     end)

# plot(dfat_mean,x=:test_correct,y=:mean,ymin=:lower,ymax=:upper,
#     xgroup=:sid,Geom.subplot_grid(Geom.errorbar,Geom.point)) # |>
#     # PDF(joinpath(plot_dir,"attend_channels.pdf"),8inch,4inch)

