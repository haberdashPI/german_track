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
using ClusterManagers

stim_info = JSON.parsefile(joinpath(stimulus_dir(),"config.json"))
eeg_files = filter(x -> occursin(r"_mcca65\.bson$",x),readdir(data_dir()))
sidfile(id) = @sprintf("eeg_response_%03d_mcca65.bson",id)

plot_dir() = joinpath(@__DIR__,"..","plots","results_$(Date(now()))")
isdir(plot_dir()) || mkdir(plot_dir())

if endswith(gethostname(),".cluster")
    addprocs(SlurmManager(length(eeg_files)), partition="CPU", t="02:00:00",
            cpus_per_task=4,enable_threaded_blas=true)
    @everywhere include(joinpath(@__DIR__,"..","util","setup.jl"))
else
    # addprocs(min(length(eeg_files),Sys.CPU_THREADS))
    # @everywhere include(joinpath(@__DIR__,"..","util","setup.jl"))
end

############################################################
# speaker analysis

########################################
# anlaysis

# TODO: we don't need this file format, we can use the 65 components directly,
# to reduce memory load.
method = OnlineMethod(window=250ms,lag=250ms,estimation_length=1.5s,γ=2e-3)
speakers = SpeakerStimMethod(encoding=ASEnvelope)

data = train_test(method,speakers,eeg_files,stim_info,
    train = "none" => no_indices,
    # progress=progress,
    test = "all_object" => row -> row.condition == "object" ?
        all_indices : no_indices,
    progress = false,
    skip_bad_trials = true)

data = DataFrame(convert(Array{OnlineResult},data))
@save joinpath(data_dir(),"test_all_online_speakers.bson") data
# @load joinpath(data_dir(),"test_all_online_speakers.bson") data

# testing...
plots = map(unique(data[data.sid .== 8,:trial])) do i
    plottrial(method,eachrow(data[(data.trial .== i) .& (data.sid .== 8),:]),
        stim_info, sidfile(data.sid[i]),raw=true)
end;

@vlplot() + vcat((hcat(pl...) for pl in Iterators.partition(plots,6))...)

indices = 1:round(Int,500ms / method.params.window)
means = by(data,[:trial,:sid,:source],norm = :norms => meanat(indices))
plot = means |>
    @vlplot(columns=4,facet={field=:sid}, title="Mean from 0 - 1 second") +
    (@vlplot(x="source:o",color=:source) +
        @vlplot(mark={:point,size=1,xOffset=-10},y=:norm,scale={zero=false}) +
        @vlplot(mark={:point, size=50, filled=true},
                y={"mean(norm)",scale={zero=false}}) +
        @vlplot(mark={:errorbar,extent=:ci},y="norm:q"))

file = joinpath(plot_dir(),"response_summary.pdf")
save(file,plot)


switch_times =
    convert(Array{Array{Float64}},stim_info["test_block_cfg"]["switch_times"])
fs = convert(Float64,stim_info["fs"])
first_switch = map(enumerate(switch_times)) do (i,times)
    target_time = stim_info["test_block_cfg"]["target_times"][i]
    if target_time > 0
        j = findlast(x -> x/fs < target_time,times)
        if j == nothing
            0
        else
            times[j]/fs
        end
    else
        missing
    end
end

means = by(data,[:trial,:sid,:source],
    norm = (:trial,:sid,:norms) => neartimes(0.0s,500.0ms,first_switch))

means |>
    @vlplot(columns=4,facet={field=:sid},
            title="Mean from first 500ms of switch before target") +
    (@vlplot(x="source:o",color=:source) +
        @vlplot(mark={:point,size=1,xOffset=-10},y=:norm,scale={zero=false}) +
        @vlplot(mark={:point, size=50, filled=true},
                y={"mean(norm)",scale={zero=false}}) +
        @vlplot(mark={:errorbar,extent=:ci},y="norm:q"))

target_times = stim_info["test_block_cfg"]["target_times"]
target_times = ifelse.(target_times .> 0,target_times,missing)
speakers = stim_info["test_block_cfg"]["trial_target_speakers"]

means = by(data,[:trial,:sid,:source],
    norm = (:trial,:sid,:norms) => neartimes(0.0s,500.0ms,target_times))
sid8 = @query(data, filter((sid == 8))) |> DataFrame
means[speakers[sound_index.(means.sid,means.trial)] .== 1,:]

means |>
    @vlplot(columns=4,facet={field=:sid},
            title="Mean from first 500ms of male target") +
    (@vlplot(x="source:o",color=:source) +
        @vlplot(mark={:point,size=1,xOffset=-10},y=:norm,scale={zero=false}) +
        @vlplot(mark={:point, size=50, filled=true},
                y={"mean(norm)",scale={zero=false}}) +
        @vlplot(mark={:errorbar,extent=:ci},y="norm:q"))

means = by(data,[:trial,:sid,:source],
    norm = (:trial,:sid,:norms) => neartimes(-500ms,0ms,target_times))
sid8 = @query(data, filter((sid == 8))) |> DataFrame
means[speakers[sound_index.(means.sid,means.trial)] .== 1,:]

means |>
    @vlplot(columns=4,facet={field=:sid},
            title="Mean 500ms before male target") +
    (@vlplot(x="source:o",color=:source) +
        @vlplot(mark={:point,size=1,xOffset=-10},y=:norm,scale={zero=false}) +
        @vlplot(mark={:point, size=50, filled=true},
                y={"mean(norm)",scale={zero=false}}) +
        @vlplot(mark={:errorbar,extent=:ci},y="norm:q"))

means = by(data,[:trial,:sid,:source],
    norm = (:trial,:sid,:norms) => neartimes(1s,1.5s,target_times))
sid8 = @query(data, filter((sid == 8))) |> DataFrame
means[speakers[sound_index.(means.sid,means.trial)] .== 1,:]

means |>
    @vlplot(columns=4,facet={field=:sid},
            title="Mean 500ms after male target") +
    (@vlplot(x="source:o",color=:source) +
        @vlplot(mark={:point,size=1,xOffset=-10},y=:norm,scale={zero=false}) +
        @vlplot(mark={:point, size=50, filled=true},
                y={"mean(norm)",scale={zero=false}}) +
        @vlplot(mark={:errorbar,extent=:ci},y="norm:q"))

####################
# testing an individual trial (to figure out why things fail)
eeg, stim_events, sid =
    load_subject(joinpath(data_dir(),sidfile(data[1,:sid])),stim_info)
fs = framerate(eeg)
stimuli = map(i -> load_speaker_mix_minus(stim_events,fs,1,i,
    encoding=:audiospect),1:5)

result = attention_marker(eegtrial(eeg,1)',stimuli...,
    framerate=framerate(eeg),
    γ=2e-3,tol=1e-2,maxit=10^2,verbose=0)

n = round(Int,uconvert(s*Hz,250ms * (fs*Hz)))
eegw = withlags(eegtrial(eeg,1)',-17:0)[1:n,:]
stimw = stimuli[1][1:n]

λ=1-n/(round(Int,uconvert(s*Hz,10s * (fs*Hz))))
result = EEGCoding.code(stimw,eegw,nothing;λ=λ,γ=1e-4,tol=1e-3,maxit=10^3,
    verbose=1)
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
@vlplot() + vcat((hcat(pl...) for pl in Iterators.partition(plots,6))...)

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
    PDF(joinpath(plot_dir(),"attend_speakers.pdf"),8inch,4inch)

############################################################
# channel analysis
online = OnlineMethod(window=250ms,lag=250ms,estimation_length=10s,γ=2e-3)
channels = ChannelStimMethod(encoding=:rms)

data = train_test(online,channels,eeg_files,stim_info,
    train = "none" => no_indices,
    test = "all_feature" => row -> row.condition == "feature" ?
        all_indices : no_indices,
    skip_bad_trials = true)

@save joinpath(data_dir(),"test_online_channels.bson") data
# @load joinpath(data_dir(),"test_online_channels.bson") data
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
    PDF(joinpath(plot_dir(),"attend_channels.pdf"),8inch,4inch)

############################################################
# first switch speaker analysis

########################################
# anlaysis

# TODO: we don't need this file format, we can use the 65 components directly,
# to reduce memory load.
method = OnlineMethod(window=250ms,lag=250ms,estimation_length=1.5s,
    γ=2e-3,tol=1e-2)
speakers = SpeakerStimMethod(encoding=:audiospect)

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

data = train_test(method,speakers,eeg_files,stim_info,
    train = "none" => no_indices,
    test = "all_object" => row -> row.condition == "object" ?
        all_indices : no_indices,
    skip_bad_trials = true)

data = DataFrame(convert(Array{OnlineResult},data))
@save joinpath(data_dir(),"test_online_first_switch_speakers_audiospect.bson") data
# @load joinpath(data_dir(),"test_online_first_switch_speakers_audiospect.bson") data

# TODO: think through this summary (there are some issues also
# noted in the top-level comments. Also worth plotting individual
# data )

# testing...
plots = map(unique(data[data.sid .== 8,:trial])) do i
    plottrial(method,eachrow(data[(data.trial .== i) .& (data.sid .== 8),:]),
        stim_info, bounds = row -> first_switch[row.sound_index],
        sidfile(data.sid[i]),raw=true)
end;

@vlplot() + vcat((hcat(pl...) for pl in Iterators.partition(plots,6))...)

means = by(data,[:trial,:sid,:source],norm = :norms => meanat(indices))
means |>
    @vlplot(columns=4,facet={field=:sid},
            title="first 500ms of 'first switch' decoder") +
    (@vlplot(x="source:o",color=:source) +
        @vlplot(mark={:point,size=1,xOffset=-10},y=:norm,scale={zero=false}) +
        @vlplot(mark={:point, size=50, filled=true},
                y={"mean(norm)",scale={zero=false}}) +
        @vlplot(mark={:errorbar,extent=:ci},y="norm:q"))

# dfat_mean = by(dfat,[:test_correct,:sid,:condition],
#     :targetattend => function(x)
#         lower,upper = dbootconf(copy(x),bootmethod=:iid,alpha=0.25)
#         (mean=mean(x),lower=lower,upper=upper)
#     end)

# plot(dfat_mean,x=:test_correct,y=:mean,ymin=:lower,ymax=:upper,
#     xgroup=:sid,Geom.subplot_grid(Geom.errorbar,Geom.point)) # |>
#     # PDF(joinpath(plot_dir(),"attend_channels.pdf"),8inch,4inch)

