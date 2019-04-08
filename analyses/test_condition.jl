include(joinpath(@__DIR__,"..","util","setup.jl"))

# - verify that everything works with CV
# - run over new data
# - look at additional conditions
# - train at target switches

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

df = DataFrame()

for eeg_file in eeg_files
    eeg, stim_events, sid = load_subject(joinpath(data_dir,eeg_file))
    for cond in unique(stim_events.condition)
        # temporary, to mimic behavior without condition iteration
        if cond != "object"
            continue
        end

        indices = findall(stim_events.condition .== cond)
        println("============================================================")
        println("Condition: $cond")

        push!(sids,fill(sid,length(indices)))

        lags = 0:round(Int,maxlag*mat"$eeg.fsample")
        for (k,(train,test)) in enumerate(folds(K,indices))
            println("----------------------------------------")
            println("Fold $k: ")
            male_model = trf_train(@sprintf("trf_$(cond)_male_sid_%03d",sid),
                eeg,stim_info,lags,train,
                name = @sprintf("Training SID %02d (Male)",sid),
                i -> load_sentence(stim_events,stim_info,i,male_index))

            fem1_model = trf_train(@sprintf("trf_$(cond)_fem1_sid_%03d",sid),
                eeg,stim_info,lags,train,
                name = @sprintf("Training SID %02d (Female 1)",sid),
                i -> load_sentence(stim_events,stim_info,i,fem1_index))

            fem2_model = trf_train(@sprintf("trf_$(cond)_fem2_sid_%03d",sid),
                eeg,stim_info,lags,train,
                name = @sprintf("Training SID %02d (Female 2)",sid),
                i -> load_sentence(stim_events,stim_info,i,fem2_index))

            # hold on a sec, we really should load the same other load_other_sentence
            # to verify that there isn't something "magic" when extract a given
            # sentence (cross-validation should also help address this)

            other_male_model = trf_train(@sprintf("trf_$(cond)_other_male_sid_%03d",sid),
                eeg,stim_info,lags,test,
                name = @sprintf("Training SID %02d (Other Male)",sid),
                i -> load_other_sentence(stim_events,stim_info,i,male_index))

            C = trf_corr(eeg,stim_info,male_model,lags,test,
                name = @sprintf("Testing SID %02d (Male)",sid),
                i -> load_sentence(stim_events,stim_info,i,male_index))
            df = vcat!(df,DataFrame(sid = sid, condition = cond, fold = k,
                speaker="male", corr = C))

            C = trf_corr(eeg,stim_info,fem1_model,lags,test,
                name = @sprintf("Testing SID %02d (Female 1)",sid),
                i -> load_sentence(stim_events,stim_info,i,fem1_index))
            df = vcat!(df,DataFrame(sid = sid, condition = cond, fold = k,
                speaker="fem1", corr = C))

            C = trf_corr(eeg,stim_info,fem2_model,lags,test,
                name = @sprintf("Testing SID %02d (Female 2)",sid),
                i -> load_sentence(stim_events,stim_info,i,fem2_index))
            df = vcat!(df,DataFrame(sid = sid, condition = cond, fold = k,
                speaker="fem2", corr = C))

            C = trf_corr(eeg,stim_info,other_male_model,lags,test,
                name = @sprintf("Testing SID %02d (Other Male)",sid),
                i -> load_sentence(stim_events,stim_info,i,other_male_index))
            df = vcat!(df,DataFrame(sid = sid, condition = cond, fold = k,
                speaker="other_male", corr = C))
        end
    end
end
save(joinpath(cache_dir,"testobj_cv.csv"),df)

alert()
