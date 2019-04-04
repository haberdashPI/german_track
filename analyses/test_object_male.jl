include(joinpath(@__DIR__,"..","util","setup.jl"))

stim_info = JSON.parse(joinpath(stimulus_dir,"config.json"))
eeg_files = readdir(joinatph(data_dir,"eeg_response*.mat"))

for eeg_file in eeg_files
    eeg,stim_evenst,sid = load_subject(eeg_file)
end
