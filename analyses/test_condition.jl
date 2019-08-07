include(joinpath(@__DIR__,"..","util","setup.jl"))

# - train on correct trials only
# - train at target switches

stim_info = JSON.parsefile(joinpath(stimulus_dir,"config.json"))
eeg_files = filter(x -> occursin(r"_mcca65\.bson$",x),readdir(data_dir))

# TODO: we don't need this file format, we can use the 65 components directly,
# to reduce memory load.

df = train_stimuli(
    StaticMethod(),
    SpeakerStimMethod(envelope_method=:rms),
    eeg_files,stim_info,
    train = "all" => all_indices,
    skip_bad_trials = true
)

save(joinpath(cache_dir(),"test_condition_rms.csv"),df)

alert()
