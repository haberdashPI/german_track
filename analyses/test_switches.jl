include(joinpath(@__DIR__,"..","util","setup.jl"))

# TODO: get only the areas before a switch to avoid areas where attentional
# tracking may be confused

stim_info = JSON.parsefile(joinpath(stimulus_dir,"config.json"))
eeg_files = filter(x -> endswith(x,".ma"*"t"),readdir(data_dir))

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

window = (-1,2)
suffix = "switches"
fs = convert(Float64,stim_info["fs"])
switch_times =
    convert(Array{Array{Float64}},stim_info["test_block_cfg"]["switch_times"])
train_bounds = remove_switches.(map(x -> x./fs,switch_times),10)

function remove_switches(switches,max_time;wait_time=0.5)
    result = Array{Tuple{Float64,Float64}}(undef,length(switches)+1)

    start = 0
    i = 0
    for switch in switches
        if start < switch
            i = i+1
            result[i] = (start,switch)
        end
        start = switch+wait_time
    end
    if start < max_time
        i = i+1
        result[i] = (start,max_time)
    end

    view(result,1:i)
end

for eeg_file in eeg_files
    global df

    eeg, stim_events, sid = load_subject(joinpath(data_dir,eeg_file))
    lags = 0:round(Int,maxlag*mat"$eeg.fsample")
    seed = hash(eeg_file)

    target_times = convert(Array{Float64},
        stim_info["test_block_cfg"]["target_times"][stim_events.sound_index])
    stim_events[:target_present] = target_times .> 0
    stim_events[:correct] = stim_events.target_present .==
        (stim_events.response .== 2)

    stim_events[:target_time] = target_times
    target_len = convert(Float64,stim_info["target_len"])

    for cond in unique(stim_events.condition)

        println("============================================================")
        println("Condition: $cond")

        indices = findall(stim_events.condition .== cond)

        male_model = trf_train(@sprintf("trf_%s_male_sid_switch_%03d",cond,sid),
            eeg,stim_info,lags,indices,
            name = @sprintf("Training SID %02d (Male): ",sid),
            bounds = train_bounds,
            group_suffix = "_"*suffix,
            i -> load_sentence(stim_events,stim_info,i,male_index))

        fem1_model = trf_train(@sprintf("trf_%s_fem1_sid_switch_%03d",cond,sid),
            eeg,stim_info,lags,indices,
            name = @sprintf("Training SID %02d (Female 1): ",sid),
            bounds = train_bounds,
            group_suffix = "_"*suffix,
            i -> load_sentence(stim_events,stim_info,i,fem1_index))

        fem2_model = trf_train(@sprintf("trf_%s_fem2_sid_switch_%03d",cond,sid),
            eeg,stim_info,lags,indices,
            name = @sprintf("Training SID %02d (Female 2): ",sid),
            bounds = train_bounds,
            group_suffix = "_"*suffix,
            i -> load_sentence(stim_events,stim_info,i,fem2_index))

        # should these also be bounded by the target?

        C = trf_corr_cv(@sprintf("trf_%s_male_sid_%03d",cond,sid),eeg,
                stim_info,male_model,lags,test,
                name = @sprintf("Testing SID %02d (Male): ",sid),
                group_suffix = "_"*suffix,
                i -> load_sentence(stim_events,stim_info,i,male_index))
        df = vcat(df,DataFrame(sid = sid, condition = cond,
                speaker="male", corr = C,
                test_correct = stim_events.correct[test]))

        C = trf_corr_cv(@sprintf("trf_%s_fem1_sid_%03d",cond,sid),eeg,
                stim_info,fem1_model,lags,test,
                name = @sprintf("Testing SID %02d (Female 1): ",sid),
                group_suffix = "_"*suffix,
                i -> load_sentence(stim_events,stim_info,i,fem1_index))
        df = vcat(df,DataFrame(sid = sid, condition = cond,
                speaker="fem1", corr = C,
                test_correct = stim_events.correct[test]))

        C = trf_corr_cv(@sprintf("trf_%s_fem2_sid_%03d",cond,sid),eeg,
                stim_info,fem2_model,lags,test,
                name = @sprintf("Testing SID %02d (Female 2): ",sid),
                group_suffix = "_"*suffix,
                i -> load_sentence(stim_events,stim_info,i,fem2_index))
            df = vcat(df,DataFrame(sid = sid, condition = cond,
                speaker="fem2", corr = C,
                test_correct = stim_events.correct[test]))

        C = trf_corr_cv(@sprintf("trf_%s_male_sid_%03d",cond,sid),eeg,
                stim_info,male_model,lags,test,
                name = @sprintf("Testing SID %02d (Other Male): ",sid),
                group_suffix = "_other_"*suffix,
                i -> load_other_sentence(stim_events,stim_info,i,male_index))
        df = vcat(df,DataFrame(sid = sid, condition = cond,
                speaker="other_male", corr = C,
                test_correct = stim_events.correct[test]))
    end
end
save(joinpath(cache_dir,"test_target.csv"),df)

alert()

