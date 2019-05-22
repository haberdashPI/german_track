include(joinpath(@__DIR__,"..","util","setup.jl"))
using Makie
using Unitful

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

@save joinpath(cache_dir(),"test_online_rms.bson") data
# @load joinpath(cache_dir(),"test_online_rms.bson") data

scene = Scene()
plottarget!(scene,method,data[5:8],stim_info,sidfile(data[1].sid))

# TODO: the target marker needs to indicate which stimulus is modulated
plotatten!(scene,method,data[5:8],raw=false)

plotswitches!(scene,method,data[5:8],stim_info,sidfile(data[1].sid))

# step 1: show the lines and bands
# step 2: show the target
# step 3: show the switches
# step 4: indicate if the resposne was correct
# step 5: show all 50 plots simultaneously

alert()
