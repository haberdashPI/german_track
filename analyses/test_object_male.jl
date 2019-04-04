include(joinpath(@__DIR__,"..","util","setup.jl"))

stim_info = JSON.parse(joinpath(stimulus_dir,"config.json"))
