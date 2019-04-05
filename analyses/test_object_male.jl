include(joinpath(@__DIR__,"..","util","setup.jl"))

stim_info = JSON.parsefile(joinpath(stimulus_dir,"config.json"))
eeg_files = filter(x -> endswith(x,".mat"),readdir(data_dir))

maxlag = 0.25
const male_index = 1
const fem1_index = 2
const fem2_index = 3

for eeg_file in eeg_files
    eeg,stim_evenst,sid = load_subject(joinpath(data_dir,eeg_file))

    lags = 0:round(Int,maxlag*eeg["fsample"])
end
