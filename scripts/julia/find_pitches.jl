using DrWatson; quickactivate(@__DIR__,"german_track")
using GermanTrack

dir = joinpath(stimulus_dir(),"mixtures","testing","mixture_components")

# NOTE: this will only work if `crepe` is installed (see README.md)
run(`crepe $dir`)
