include(joinpath(@__DIR__,"..","util","setup.jl"))

# - train on correct trials only
# - train at target switches

stim_info = JSON.parsefile(joinpath(stimulus_dir,"config.json"))
eeg_files = filter(x -> occursin(r"_mcca65\.bson$",x),readdir(data_dir))

# TODO: we don't need this file format, we can use the 65 components directly,
# to reduce memory load.

df = trf_train_speakers("cleaned",eeg_files,stim_info,
    train = "condition" => all_indices,
    test = "condition" => all_indices,
    envelope_method = :audiospect,
    skip_bad_trials = true)

save(joinpath(cache_dir(),"test_condition.csv"),df)

alert()
