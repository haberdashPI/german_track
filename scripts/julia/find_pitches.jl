using DrWatson; quickactivate(@__DIR__,"german_track")
include(joinpath(srcdir(),"julia","setup.jl"))

dir = joinpath(raw_stim_dir(),"sentences")
resultdir = joinpath(stimulus_dir(),"pitches")
isdir(resultdir) || mkdir(resultdir)

# NOTE: this will only work if `crepe` is installed (see README.md)
run(`crepe $dir -o $resultdir`)
