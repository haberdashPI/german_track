include(joinpath(@__DIR__,"..","util","setup.jl"))

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
for eeg_file in eeg_files
    eeg, stim_evenst, sid = load_subject(joinpath(data_dir,eeg_file))
    indices = find(stim_events.condition .== "object")

    push!(sids,fill(sid,length(indices))

    lags = 0:round(Int,maxlag*eeg["fsample"])

    male_model = trf_train(@sprintf('trf_object_male_sid_%03d',i),
        eeg,stim_info,lags,indices,
        i -> load_sentence(stim_events,stim_info,i,male_index))

    fem1_model = trf_train(@sprintf('trf_object_fem1_sid_%03d',i),
        eeg,stim_info,lags,indices,
        i -> load_sentence(stim_events,stim_info,i,fem1_index))

    fem2_model = trf_train(@sprintf('trf_object_fem2_sid_%03d',i),
        eeg,stim_info,lags,indices,
        i -> load_sentence(stim_events,stim_info,i,fem2_index))

    other_male_model = trf_train(@sprintf('trf_object_other_male_sid_%03d',i),
        eeg,stim_info,lags,indices,
        i -> load_other_sentence(stim_events,stim_info,i,male_index))

    push!(male_C,trf_corr(eeg,stim_info,male_model,lags,indices,
                    i -> load_sentence(stim_events,stim_info,i,male_index)))

    push!(fem1_C,trf_corr(eeg,stim_info,fem1_model,lags,indices,
                    i -> load_sentence(stim_events,stim_info,i,fem1_index)))

    push!(fem2_C,trf_corr(eeg,stim_info,fem2_model,lags,indices,
                    i -> load_sentence(stim_events,stim_info,i,fem2_index)))

    push!(other_male_C,trf_corr(eeg,stim_info,other_male_model,lags,indices,
                    i -> load_other_sentence(stim_events,stim_info,i,male_index)))
end

save("testobj.csv",DataFrame(
    male_C=vcat(male_C...),
    fem1_C=vcat(fem1_C...),
    fem2_C=vcat(fem2_C...),
    other_male_C=vcat(other_male_C...),
    sid=repeat()))
