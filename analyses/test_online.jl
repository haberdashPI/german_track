include(joinpath(@__DIR__,"..","util","setup.jl"))
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
data = convert(Array{OnlineResult},data)

main = Scene();
trials = map(Iterators.take(groupby(DataFrame(data),:trial),24)) do results
   plottrial(method,eachrow(results),stim_info,sidfile(data[1].sid))
end;
trials = vbox(map(x -> hbox(x...),Iterators.partition(trials,6))...);
Makie.save("online_test.png",trials);

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

# things to count up:
# -

alert()
