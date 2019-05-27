include(joinpath(@__DIR__,"..","util","setup.jl"))
using Gadfly
using Makie
using Unitful
using DataKnots
using Tables

# - train on correct trials only
# - train at target switches

stim_info = JSON.parsefile(joinpath(stimulus_dir,"config.json"))
eeg_files = filter(x -> occursin(r"_mcca65\.bson$",x),readdir(data_dir))
sidfile(id) = @sprintf("eeg_response_%03d_mcca65.bson",id)

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


main = Scene();
sid8 = @query(data, filter((sid == 8) & (trial <= 75))) |> DataFrame

trials = []
for trial in groupby(sid8,:trial)
   push!(trials,plottrial(method,eachrow(trial),stim_info,sidfile(data.sid[1])))
end

Makie.save("online_test.png",vbox(map(x -> hbox(x...),
    Iterators.partition(trials,6))...));

# stim_events, = events_for_eeg(sidfile(row.sid),stim_info)

dfat = by(data,:sid) do dfsid
    stim_events, = events_for_eeg(sidfile(dfsid.sid[1]),stim_info)
    dfsid[:targetattend] = map(row -> targetattend(row,stim_events,stim_info),eachrow(dfsid))
    dfsid
end


# + step 1: show the lines and bands
# + step 2: show the target
# + step 3: show the switches
# + step 4: indicate if the resposne was correct
# + step 5: show all 50 plots simultaneously

# parameters to mess around with & tune:
# sparsity \gamma
# lag
# estimation_length
# smoothness parameters

# most of these don't seem to make a large difference in the outcomes, at least
# when looking at a single trial; even sparsity doesn't matter too much as long
# as some exists. I haven't messed around with lag though. As long as the
# values are somewhat "reasonable" about the same result is found

# things to count up:
# - does the attended speaker help us predict correct responses?
# - does the attended speaker depend on the switches?
# - does the "buildup-up curve" help us predict the behavioral data?

# Goal: do we have any evidence that the decoding is real?
# that is, does it correspond to behavior somehow

# Do the same analysis for:
# 1. mixture envelope (with and without the target speaker?? may not matter)
# 2. the audiospect envelope
# 3. left vs. right on the feature condition
# 4. both LvR and Speakers for the global condition

alert()
