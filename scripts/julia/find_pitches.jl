using DrWatson; quickactivate(@__DIR__,"german_track")
using GermanTrack

dir = joinpath(stimulus_dir(),"mixtures","testing","mixture_components")

cd(projectdir())
run(`pipenv run crepe $dir`)
