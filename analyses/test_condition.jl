include(joinpath(@__DIR__,"..","util","setup.jl"))

# - train on correct trials only
# - train at target switches

stim_info = JSON.parsefile(joinpath(stimulus_dir,"config.json"))
eeg_files = filter(x -> occursin(r"_cleaned\.mat$",x),readdir(data_dir))

df = trf_train_speakers("cleaned",eeg_files,stim_info,
    train = "condition" => all_indices,
    test = "condition" => all_indices,
    skip_bad_trials = true)

save(joinpath(cache_dir,"test_switches.csv"),df)

alert()
