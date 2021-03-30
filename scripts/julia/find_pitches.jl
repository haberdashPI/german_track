using DrWatson; quickactivate(@__DIR__,"german_track")
using GermanTrack, Underscores

dir = joinpath(stimulus_dir(),"mixtures","testing","mixture_component_channels")
cd(projectdir())
run(`pipenv run crepe $dir`)

# move results to a new directory
resultdir = mkpath(joinpath(stimulus_dir(), "mixtures","testing","mixture_component_pitches"))
for file in @_ readdir(dir) |> filter(endswith(_, ".f0.csv"), __)
    mv(joinpath(dir, file), joinpath(resultdir, file))
end
