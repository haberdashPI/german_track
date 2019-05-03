include(joinpath(@__DIR__,"..","util","setup.jl"))

stim_info = JSON.parsefile(joinpath(stimulus_dir,"config.json"))
eeg_files = filter(x -> occursin(r"_mcca65\.bson$",x),readdir(data_dir))

file = eeg_files[1]
eegjl, stim_events, sid = load_subject(joinpath(data_dir,file),stim_info)
eegmat, = load_subject(joinpath(data_dir,replace(file,"bson"=>"mat")),stim_info)
stim = load_sentence(stim_events,stim_info,1,1)

modeljl = find_trf(stim,eegjl,1,-1,0:17,"Shrinkage")
modelmat = find_trf(stim,eegmat,1,-1,0:17,"Shrinkage")

ŷ_jl = predict_trf(-1,eegtrial(eegjl,1),modeljl,0:17,"Shrinkage")
ŷ_mat = predict_trf(-1,eegtrial(eegmat,1),modelmat,0:17,"Shrinkage")

maximum(abs,ŷ_mat .- ŷ_jl)

@btime find_trf(stim,eegjl,1,-1,0:17,"Shrinkage")
@btime find_trf(stim,eegmat,1,-1,0:17,"Shrinkage")
