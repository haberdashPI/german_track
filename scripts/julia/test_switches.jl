
include(joinpath(@__DIR__,"..","util","setup.jl"))

# TODO: get only the areas before a switch to avoid areas where attentional
# tracking may be confused

stim_info = JSON.parsefile(joinpath(stimulus_dir(),"config.json"))
eeg_files = filter(x -> occursin(r"_cleaned\.mat$",x),readdir(data_dir()))

df = DataFrame(sid = Int[],condition = String[], speaker = String[],
        corr = Float64[],test_correct = Bool[])

fs = convert(Float64,stim_info["fs"])
switch_times =
    convert(Array{Array{Float64}},stim_info["test_block_cfg"]["switch_times"])
switch_bounds = only_switches.(map(x -> x./fs,switch_times),10)

df = decode_speakers("",eeg_files,stim_info,
    train = "clean_switch_only" => row -> switch_bounds[row.sound_index],
    test = "clean_switch_only" => row -> switch_bounds[row.sound_index],
    skip_bad_trials=true)

save(joinpath(cache_dir(),"test_only_switches.csv"))
