include(joinpath(@__DIR__,"..","util","setup.jl"))

# - train at correct targets

stim_info = JSON.parsefile(joinpath(stimulus_dir,"config.json"))
eeg_files = filter(x -> occursin(r"_mcca05\.mat$",x),readdir(data_dir))

df = trf_train_speakers("",eeg_files,stim_info,
    train = "mcca05_correct" =>
        row -> row.correct ? all_indices : no_indices,
    test = "mcca05_correct" => all_indices,
    skip_bad_trials = true)

save(joinpath(cache_dir,"test_correct.csv"),df)

alert()
