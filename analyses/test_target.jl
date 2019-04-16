include(joinpath(@__DIR__,"..","util","setup.jl"))

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
suffix = "correct"
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

        test = findall((stim_events.condition .== cond) .&
            (stim_events.target_time .> 0))
        train = findall((stim_events.condition .== cond) .&
            (stim_events.target_time .> 0) .& (stim_events.correct))

        target_bounds = tuple.(stim_events.target_time .+ window[1]*target_len,
            stim_events.target_time .+ window[2]*target_len)

        male_model = trf_train(@sprintf("trf_%s_male_target_sid_%03d",cond,sid),
            eeg,stim_info,lags,train,
            name = @sprintf("Training SID %02d (Male): ",sid),
            bounds = target_bounds,
            group_suffix = "_"*suffix,
            i -> load_sentence(stim_events,stim_info,i,male_index))

        fem1_model = trf_train(@sprintf("trf_%s_fem1_target_sid_%03d",cond,sid),
            eeg,stim_info,lags,train,
            name = @sprintf("Training SID %02d (Female 1): ",sid),
            bounds = target_bounds,
            group_suffix = "_"*suffix,
            i -> load_sentence(stim_events,stim_info,i,fem1_index))

        fem2_model = trf_train(@sprintf("trf_%s_fem2_target_sid_%03d",cond,sid),
            eeg,stim_info,lags,train,
            name = @sprintf("Training SID %02d (Female 2): ",sid),
            bounds = target_bounds,
            group_suffix = "_"*suffix,
            i -> load_sentence(stim_events,stim_info,i,fem2_index))

        # should these also be bounded by the target?

        C = trf_corr_cv(@sprintf("trf_%s_male_target_sid_%03d",cond,sid),eeg,
                stim_info,male_model,lags,test,
                name = @sprintf("Testing SID %02d (Male): ",sid),
                group_suffix = "_"*suffix,
                bounds = target_bounds,
                i -> load_sentence(stim_events,stim_info,i,male_index))
        df = vcat(df,DataFrame(sid = sid, condition = cond,
                speaker="male", corr = C,
                test_correct = stim_events.correct[test]))

        C = trf_corr_cv(@sprintf("trf_%s_fem1_target_sid_%03d",cond,sid),eeg,
                stim_info,fem1_model,lags,test,
                name = @sprintf("Testing SID %02d (Female 1): ",sid),
                group_suffix = "_"*suffix,
                bounds = target_bounds,
                i -> load_sentence(stim_events,stim_info,i,fem1_index))
        df = vcat(df,DataFrame(sid = sid, condition = cond,
                speaker="fem1", corr = C,
                test_correct = stim_events.correct[test]))

        C = trf_corr_cv(@sprintf("trf_%s_fem2_target_sid_%03d",cond,sid),eeg,
                stim_info,fem2_model,lags,test,
                name = @sprintf("Testing SID %02d (Female 2): ",sid),
                group_suffix = "_"*suffix,
                bounds = target_bounds,
                i -> load_sentence(stim_events,stim_info,i,fem2_index))
            df = vcat(df,DataFrame(sid = sid, condition = cond,
                speaker="fem2", corr = C,
                test_correct = stim_events.correct[test]))

        C = trf_corr_cv(@sprintf("trf_%s_male_target_sid_%03d",cond,sid),eeg,
                stim_info,male_model,lags,test,
                name = @sprintf("Testing SID %02d (Other Male): ",sid),
                group_suffix = "_other_"*suffix,
                bounds = target_bounds,
                i -> load_other_sentence(stim_events,stim_info,i,male_index))
        df = vcat(df,DataFrame(sid = sid, condition = cond,
                speaker="other_male", corr = C,
                test_correct = stim_events.correct[test]))
    end
end
save(joinpath(cache_dir,"test_target.csv"),df)

alert()

