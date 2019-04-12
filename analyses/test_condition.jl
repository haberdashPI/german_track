include(joinpath(@__DIR__,"..","util","setup.jl"))

# - verify that everything works with CV
# - run over new data
# - look at additional conditions
# - train at target switches

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
        corr = Float64[])

for eeg_file in eeg_files
    global df

    eeg, stim_events, sid = load_subject(joinpath(data_dir,eeg_file))
    lags = 0:round(Int,maxlag*mat"$eeg.fsample")
    seed = hash(eeg_file)
    # test and train generates the same random sequence

    for cond in unique(stim_events.condition)

        indices = findall(stim_events.condition .== cond)
        println("============================================================")
        println("Condition: $cond")

        push!(sids,fill(sid,length(indices)))

        male_model = trf_train(@sprintf("trf_%s_male_sid_%03d",cond,sid),
            eeg,stim_info,lags,indices,
            name = @sprintf("Training SID %02d (Male): ",sid),
            i -> load_sentence(stim_events,stim_info,i,male_index))

        fem1_model = trf_train(@sprintf("trf_%s_fem1_sid_%03d",cond,sid),
            eeg,stim_info,lags,indices,
            name = @sprintf("Training SID %02d (Female 1): ",sid),
            i -> load_sentence(stim_events,stim_info,i,fem1_index))

        fem2_model = trf_train(@sprintf("trf_%s_fem2_sid_%03d",cond,sid),
            eeg,stim_info,lags,indices,
            name = @sprintf("Training SID %02d (Female 2): ",sid),
            i -> load_sentence(stim_events,stim_info,i,fem2_index))

        C = trf_corr_cv(@sprintf("trf_%s_male_sid_%03d",cond,sid),eeg,
                stim_info,male_model,lags,indices,
                name = @sprintf("Testing SID %02d (Male): ",sid),
                i -> load_sentence(stim_events,stim_info,i,male_index))
        df = vcat(df,DataFrame(sid = sid, condition = cond,
                speaker="male", corr = C))

        C = trf_corr_cv(@sprintf("trf_%s_fem1_sid_%03d",cond,sid),eeg,
                stim_info,fem1_model,lags,indices,
                name = @sprintf("Testing SID %02d (Female 1): ",sid),
                i -> load_sentence(stim_events,stim_info,i,fem1_index))
        df = vcat(df,DataFrame(sid = sid, condition = cond,
                speaker="fem1", corr = C))

        C = trf_corr_cv(@sprintf("trf_%s_male_sid_%03d",cond,sid),eeg,
                stim_info,fem2_model,lags,indices,
                name = @sprintf("Testing SID %02d (Female 2): ",sid),
                i -> load_sentence(stim_events,stim_info,i,fem2_index))
            df = vcat(df,DataFrame(sid = sid, condition = cond,
                speaker="fem2", corr = C))

        C = trf_corr_cv(@sprintf("trf_%s_male_sid_%03d",cond,sid),eeg,
                stim_info,male_model,lags,indices,
                name = @sprintf("Testing SID %02d (Other Male): ",sid),
                i -> load_other_sentence(stim_events,stim_info,i,male_index))
        df = vcat(df,DataFrame(sid = sid, condition = cond,
                speaker="other_male", corr = C))
    end
end
save(joinpath(cache_dir,"testcond.csv"),df)

alert()
