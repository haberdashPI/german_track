include(joinpath(@__DIR__,"..","util","setup.jl"))

# TODO: get only the areas before a switch to avoid areas where attentional
# tracking may be confused

stim_info = JSON.parsefile(joinpath(stimulus_dir,"config.json"))
eeg_files = filter(x -> endswith(x,".mat"),readdir(data_dir))

maxlag = 0.25
const male_index = 1
const fem1_index = 2
const fem2_index = 3

male_C = []
fem1_C = []
fem2_C = []
other_male_C = []
sids = []
K = 10

df = DataFrame(sid = Int[],condition = String[], speaker = String[],
        corr = Float64[],test_correct = Bool[])

fs = convert(Float64,stim_info["fs"])
switch_times =
    convert(Array{Array{Float64}},stim_info["test_block_cfg"]["switch_times"])
switch_bounds = remove_switches.(map(x -> x./fs,switch_times),10)

df = trf_train_speakers(stim_info,
    train = "noswitches" => row -> switch_bounds[row.sound_index],
    test = "noswitches" => row -> switch_bounds[row.sound_index])
save(joinpath(cache_dir,"test_switches.csv"),df)

alert()

