include(joinpath(@__DIR__,"..","util","setup.jl"))

# - train on correct trials only
# - train at target switches

stim_info = JSON.parsefile(joinpath(stimulus_dir,"config.json"))
eeg_files = filter(x -> occursin(r"_mcca65\.bson$",x),readdir(data_dir))

# TODO: we don't need this file format, we can use the 65 components directly,
# to reduce memory load.

df = trf_train_speakers("",eeg_files,stim_info,
    train = "rms_condition" => all_indices,
    test = "rms_condition" => all_indices,
    envelope_method = :rms,
    skip_bad_trials = true)

save(joinpath(cache_dir(),"test_condition_rms.csv"),df)

alert()
