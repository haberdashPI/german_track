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

# TODO: speed up caching, (do so after cross-validation) by storing aggregate
# model

male_C = []
fem1_C = []
fem2_C = []
other_male_C = []
sids = []

# TODO: once this is basically working, convert to an all julia implementation:
# I don't really need the telluride toolbox to do what I'm doing right now

for eeg_file in eeg_files
    eeg, stim_events, sid = load_subject(joinpath(data_dir,eeg_file))
    indices = findall(stim_events.condition .== "object")

    push!(sids,fill(sid,length(indices)))

    lags = 0:round(Int,maxlag*mat"$eeg.fsample")

    male_model = trf_train(@sprintf("trf_object_male_sid_%03d",sid),
        eeg,stim_info,lags,indices,
        name = @sprintf("Training SID %02d (Male)",sid),
        i -> load_sentence(stim_events,stim_info,i,male_index))

    fem1_model = trf_train(@sprintf("trf_object_fem1_sid_%03d",sid),
        eeg,stim_info,lags,indices,
        name = @sprintf("Training SID %02d (Female 1)",sid),
        i -> load_sentence(stim_events,stim_info,i,fem1_index))

    fem2_model = trf_train(@sprintf("trf_object_fem2_sid_%03d",sid),
        eeg,stim_info,lags,indices,
        name = @sprintf("Training SID %02d (Female 2)",sid),
        i -> load_sentence(stim_events,stim_info,i,fem2_index))

    # hold on a sec, we really should load the same other load_other_sentence
    # to verify that there isn't something "magic" when extract a given
    # sentence (cross-validation should also help address this)

    other_male_model = trf_train(@sprintf("trf_object_other_male_sid_%03d",sid),
        eeg,stim_info,lags,indices,
        name = @sprintf("Training SID %02d (Other Male)",sid),
        i -> load_other_sentence(stim_events,stim_info,i,male_index))

    push!(male_C,trf_corr(eeg,stim_info,male_model,lags,indices,
                    name = @sprintf("Testing SID %02d (Male)",sid),
                    i -> load_sentence(stim_events,stim_info,i,male_index)))

    push!(fem1_C,trf_corr(eeg,stim_info,fem1_model,lags,indices,
                    name = @sprintf("Testing SID %02d (Female 1)",sid),
                    i -> load_sentence(stim_events,stim_info,i,fem1_index)))

    push!(fem2_C,trf_corr(eeg,stim_info,fem2_model,lags,indices,
                    name = @sprintf("Testing SID %02d (Female 2)",sid),
                    i -> load_sentence(stim_events,stim_info,i,fem2_index)))

    push!(other_male_C,trf_corr(eeg,stim_info,other_male_model,lags,indices,
                    name = @sprintf("Testing SID %02d (Other Male)",sid),
                    i -> load_other_sentence(stim_events,stim_info,i,male_index)))
end

save(joinpath(cache_dir,"testobj.csv"),DataFrame(
    male_C=vcat(male_C...),
    fem1_C=vcat(fem1_C...),
    fem2_C=vcat(fem2_C...),
    other_male_C=vcat(other_male_C...),
    sid=vcat(sids...)))

alert()
