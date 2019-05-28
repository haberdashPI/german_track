include(joinpath(@__DIR__,"..","util","setup.jl"))
using Gadfly, Cairo, Fontconfig
using DependentBootstrap
using Makie
using Unitful
using DataKnots
using Tables

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
data = train_speakers(method,"",eeg_files,stim_info,
    train = "rms_online" => no_indices,
    test = "rms_online" => row -> row.condition == "object" ?
        all_indices : no_indices,
    envelope_method = :rms,
    skip_bad_trials = true)

@save joinpath(data_dir,"test_online_rms.bson") data
# @load joinpath(data_dir,"test_online_rms.bson") data
data = DataFrame(convert(Array{OnlineResult},data))

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

# stim_events, = events_for_eeg(sidfile(row.sid),stim_info)

########################################
# summary plot

dfat = by(data,:sid) do dfsid
    stim_events, = events_for_eeg(sidfile(dfsid.sid[1]),stim_info)
    by(data,:trial) do dftrial
        attend = targetattend(eachrow(dftrial),stim_events,stim_info,
            ustrip(uconvert(Hz,1/method.params.window)))
        DataFrame(
            targetattend = attend,
            test_correct = dftrial.test_correct[1],
            condition = dftrial.condition[1]
        )
    end
end

# TODO: group by condition as well as sid
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

method = OnlineMethod(window=250ms,lag=250ms,estimation_length=10s,γ=2e-3)
data = train_channels(method,"",eeg_files,stim_info,
    train = "rms_online" => no_indices,
    test = "rms_online" => row -> row.condition == "object" ?
        all_indices : no_indices,
    envelope_method = :rms,
    skip_bad_trials = true)

# things to count up:
# - does the attended speaker depend on the switches?
# - does the "buildup-up curve" help us predict the behavioral data?

# Do the same analysis for:
# 1. mixture envelope (with and without the target speaker?? may not matter)
# 2. the audiospect envelope
# 3. left vs. right on the feature condition
# 4. both LvR and Speakers for the global condition

alert()