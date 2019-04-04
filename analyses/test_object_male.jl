include(joinpath(@__DIR__,"..","util","setup.jl"))

stim_info = JSON.parse(joinpath(stimulus_dir,"config.json"))
eeg_files = filter(occursin(r"\.m$"),readdir(data_dir))

for eeg_file in eeg_files
    eeg,stim_evenst,sid = load_subject(joinpath(data_dir,eeg_file))
end
