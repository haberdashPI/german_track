# add DrWatson to root environment
using Pkg; pkg"add DrWatson"
using DrWatson

mv("Project.toml","Project.toml.backup")
cp("Project.toml.install","Project.toml")
cp("Manifest.toml.local","Manifest.toml")
quickactivate(@__DIR__,"german_track")

pkg"dev src/julia/EEGCoding"
pkg"dev src/julia/GermanTrack"
Pkg.instantiate()
