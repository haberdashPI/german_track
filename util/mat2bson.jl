include(joinpath(@__DIR__,"..","util","setup.jl"))

eeg_files = filter(x -> occursin(r"[a-z_][0-9]{2}\.mat$",x),readdir(data_dir))

test_dir = joinpath(homedir(),"Documents","test")
isdir(test_dir) || mkdir(test_dir)

@showprogress for file in eeg_files
    mf = MatFile(joinpath(data_dir,file))
    data = get_variable(mf,:dat)
    close(mf)

    data = clean_eeg!(data)
    bson(joinpath(test_dir,replace(file,r"\.mat$" => ".bson")),data=data)
end
