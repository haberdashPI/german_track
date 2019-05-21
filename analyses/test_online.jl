include(joinpath(@__DIR__,"..","util","setup.jl"))

# - train on correct trials only
# - train at target switches

stim_info = JSON.parsefile(joinpath(stimulus_dir,"config.json"))
eeg_files = filter(x -> occursin(r"_mcca65\.bson$",x),readdir(data_dir))

# TODO: we don't need this file format, we can use the 65 components directly,
# to reduce memory load.

# start with a limited data subset (subj 1, object only)
result = train_speakers(OnlineMethod(),"",eeg_files[1:1],stim_info,
    train = "rms_online" => no_indices,
    test = "rms_online" => row ->
        row.condition == "object" ? all_indices : no_indices,
    envelope_method = :rms,
    skip_bad_trials = true)

save(joinpath(cache_dir(),"test_online_rms.bson"),result)

alert()
