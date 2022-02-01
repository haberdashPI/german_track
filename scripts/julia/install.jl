# add DrWatson to root environment
using Pkg; Pkg.add("DrWatson")
using DrWatson

mv("Project.toml","Project.toml.backup")
cp("Project.toml.install","Project.toml")
cp("Manifest.toml.local","Manifest.toml")
quickactivate(@__DIR__,"german_track")

Pkg.develop(path="src/julia/EEGCoding")
Pkg.develop(path="src/julia/GermanTrack")
Pkg.instantiate()
